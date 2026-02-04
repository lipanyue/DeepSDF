// Copyright 2004-present Facebook. All Rights Reserved.

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#include <pangolin/geometry/geometry.h>
#include <pangolin/geometry/glgeometry.h>
#include <pangolin/gl/gl.h>
#include <pangolin/pangolin.h>

#include <CLI/CLI.hpp>
#include <cnpy.h>

#include "Utils.h"

extern pangolin::GlSlProgram GetShaderProgram();

void SampleFromSurface(
    pangolin::Geometry& geom,
    std::vector<Eigen::Vector3f>& surfpts,
    int num_sample) {
  float total_area = 0.0f;

  std::vector<float> cdf_by_area;

  std::vector<Eigen::Vector3i> linearized_faces;

  for (const auto& object : geom.objects) {
    auto it_vert_indices = object.second.attributes.find("vertex_indices");
    if (it_vert_indices != object.second.attributes.end()) {
      pangolin::Image<uint32_t> ibo =
          pangolin::get<pangolin::Image<uint32_t>>(it_vert_indices->second);

      for (int i = 0; i < ibo.h; ++i) {
        linearized_faces.emplace_back(ibo(0, i), ibo(1, i), ibo(2, i));
      }
    }
  }

  pangolin::Image<float> vertices =
      pangolin::get<pangolin::Image<float>>(geom.buffers["geometry"].attributes["vertex"]);

  for (const Eigen::Vector3i& face : linearized_faces) {
    float area = TriangleArea(
        (Eigen::Vector3f)Eigen::Map<Eigen::Vector3f>(vertices.RowPtr(face(0))),
        (Eigen::Vector3f)Eigen::Map<Eigen::Vector3f>(vertices.RowPtr(face(1))),
        (Eigen::Vector3f)Eigen::Map<Eigen::Vector3f>(vertices.RowPtr(face(2))));

    if (std::isnan(area)) {
      area = 0.f;
    }

    total_area += area;

    if (cdf_by_area.empty()) {
      cdf_by_area.push_back(area);

    } else {
      cdf_by_area.push_back(cdf_by_area.back() + area);
    }
  }

  std::random_device seeder;
  std::mt19937 generator(seeder());
  std::uniform_real_distribution<float> rand_dist(0.0, total_area);

  while ((int)surfpts.size() < num_sample) {
    float tri_sample = rand_dist(generator);
    std::vector<float>::iterator tri_index_iter =
        lower_bound(cdf_by_area.begin(), cdf_by_area.end(), tri_sample);
    int tri_index = tri_index_iter - cdf_by_area.begin();

    const Eigen::Vector3i& face = linearized_faces[tri_index];

    surfpts.push_back(SamplePointFromTriangle(
        Eigen::Map<Eigen::Vector3f>(vertices.RowPtr(face(0))),
        Eigen::Map<Eigen::Vector3f>(vertices.RowPtr(face(1))),
        Eigen::Map<Eigen::Vector3f>(vertices.RowPtr(face(2)))));
  }
}

void SampleSDFNearSurface(
    KdVertexListTree& kdTree,
    std::vector<Eigen::Vector3f>& vertices,
    std::vector<Eigen::Vector3f>& xyz_surf,
    std::vector<Eigen::Vector3f>& normals,
    std::vector<Eigen::Vector3f>& xyz,
    std::vector<float>& sdfs,
    int num_rand_samples,
    float variance,
    float second_variance,
    float bounding_cube_dim,
    int num_votes) {
  float stdv = sqrt(variance);

  std::random_device seeder;
  std::mt19937 generator(seeder());
  std::uniform_real_distribution<float> rand_dist(0.0, 1.0);
  std::vector<Eigen::Vector3f> xyz_used;
  std::vector<Eigen::Vector3f> second_samples;

  std::random_device rd;
  std::mt19937 rng(rd());
  std::uniform_int_distribution<int> vert_ind(0, vertices.size() - 1);
  std::normal_distribution<float> perterb_norm(0, stdv);
  std::normal_distribution<float> perterb_second(0, sqrt(second_variance));

  // 表面附近采样
  for (unsigned int i = 0; i < xyz_surf.size(); i++) {
    Eigen::Vector3f surface_p = xyz_surf[i];
    Eigen::Vector3f samp1 = surface_p;
    Eigen::Vector3f samp2 = surface_p;

    for (int j = 0; j < 3; j++) {
      // ✅ 修复：限制噪声范围，确保点在[-1, 1]内
      float noise1 = perterb_norm(rng);
      float noise2 = perterb_second(rng);
      
      // 应用噪声，但确保不超出[-0.95, 0.95]
      samp1[j] = std::clamp(samp1[j] + noise1, -0.95f, 0.95f);
      samp2[j] = std::clamp(samp2[j] + noise2, -0.95f, 0.95f);
    }

    xyz.push_back(samp1);
    xyz.push_back(samp2);
  }

  // 均匀空间采样
  for (int s = 0; s < (int)(num_rand_samples); s++) {
    // ✅ 修复：采样范围改为[-0.9, 0.9]，为噪声留出空间
    xyz.push_back(Eigen::Vector3f(
        (rand_dist(generator) * 1.8 - 0.9),
        (rand_dist(generator) * 1.8 - 0.9),
        (rand_dist(generator) * 1.8 - 0.9)));
  }

  // 计算每个点的SDF
  for (int s = 0; s < (int)xyz.size(); s++) {
    Eigen::Vector3f samp_vert = xyz[s];
    std::vector<int> cl_indices(num_votes);
    std::vector<float> cl_distances(num_votes);
    kdTree.knnSearch(samp_vert.data(), num_votes, cl_indices.data(), cl_distances.data());

    int num_pos = 0;
    float sdf = 0.0f;

    // 使用最近点计算SDF
    for (int ind = 0; ind < num_votes; ind++) {
      uint32_t cl_ind = cl_indices[ind];
      Eigen::Vector3f cl_vert = vertices[cl_ind];
      Eigen::Vector3f ray_vec = samp_vert - cl_vert;
      float ray_vec_leng = ray_vec.norm();

      if (ind == 0) {
        // ✅ 修复：始终使用欧氏距离作为距离值
        sdf = ray_vec_leng;
      }

      // 统计法线方向（用于确定符号）
      if (ray_vec_leng > 1e-6f) {
        float d = normals[cl_ind].dot(ray_vec / ray_vec_leng);
        if (d > 0)
          num_pos++;
      }
    }

    // all or nothing , else ignore the point
    if ((num_pos == 0) || (num_pos == num_votes)) {
      xyz_used.push_back(samp_vert);
      
      // ✅ 修复：根据法线投票确定符号
      // 内部为负，外部为正
      if (num_pos == 0) {
        sdf = -std::abs(sdf);  // 内部点
      } else {
        sdf = std::abs(sdf);   // 外部点
      }
      sdfs.push_back(sdf);
    }
  }

  // ✅ 修复：再次裁剪确保所有点在[-1, 1]范围内
  for (auto& point : xyz_used) {
    point.x() = std::clamp(point.x(), -1.0f, 1.0f);
    point.y() = std::clamp(point.y(), -1.0f, 1.0f);
    point.z() = std::clamp(point.z(), -1.0f, 1.0f);
  }
  
  xyz = xyz_used;
}

void writeSDFToNPY(
    std::vector<Eigen::Vector3f>& xyz,
    std::vector<float>& sdfs,
    std::string filename) {
  unsigned int num_vert = xyz.size();
  std::vector<float> data(num_vert * 4);
  int data_i = 0;

  for (unsigned int i = 0; i < num_vert; i++) {
    Eigen::Vector3f v = xyz[i];
    float s = sdfs[i];

    for (int j = 0; j < 3; j++)
      data[data_i++] = v[j];
    data[data_i++] = s;
  }

  cnpy::npy_save(filename, &data[0], {(long unsigned int)num_vert, 4}, "w");
}

void writeSDFToNPZ(
    std::vector<Eigen::Vector3f>& xyz,
    std::vector<float>& sdfs,
    std::string filename,
    bool print_num = false) {
  unsigned int num_vert = xyz.size();
  std::vector<float> pos;
  std::vector<float> neg;

  for (unsigned int i = 0; i < num_vert; i++) {
    Eigen::Vector3f v = xyz[i];
    float s = sdfs[i];

    if (s < 0) {
      for (int j = 0; j < 3; j++)
        pos.push_back(v[j]);
      pos.push_back(s);
    } else {
      for (int j = 0; j < 3; j++)
        neg.push_back(v[j]);
      neg.push_back(s);
    }
  }

  cnpy::npz_save(filename, "pos", &pos[0], {(long unsigned int)(pos.size() / 4.0), 4}, "w");
  cnpy::npz_save(filename, "neg", &neg[0], {(long unsigned int)(neg.size() / 4.0), 4}, "a");
  if (print_num) {
    std::cout << "pos num: " << pos.size() / 4.0 << std::endl;
    std::cout << "neg num: " << neg.size() / 4.0 << std::endl;
  }
}

void writeSDFToPLY(
    std::vector<Eigen::Vector3f>& xyz,
    std::vector<float>& sdfs,
    std::string filename,
    bool neg_only = true,
    bool pos_only = false) {
  int num_verts;
  if (neg_only) {
    num_verts = 0;
    for (int i = 0; i < (int)sdfs.size(); i++) {
      float s = sdfs[i];
      if (s <= 0)
        num_verts++;
    }
  } else if (pos_only) {
    num_verts = 0;
    for (int i = 0; i < (int)sdfs.size(); i++) {
      float s = sdfs[i];
      if (s >= 0)
        num_verts++;
    }
  } else {
    num_verts = xyz.size();
  }

  std::ofstream plyFile;
  plyFile.open(filename);
  plyFile << "ply\n";
  plyFile << "format ascii 1.0\n";
  plyFile << "element vertex " << num_verts << "\n";
  plyFile << "property float x\n";
  plyFile << "property float y\n";
  plyFile << "property float z\n";
  plyFile << "property uchar red\n";
  plyFile << "property uchar green\n";
  plyFile << "property uchar blue\n";
  plyFile << "end_header\n";

  for (int i = 0; i < (int)sdfs.size(); i++) {
    Eigen::Vector3f v = xyz[i];
    float sdf = sdfs[i];
    bool neg = (sdf <= 0);
    bool pos = (sdf >= 0);
    if (neg)
      sdf = -sdf;
    int sdf_i = std::min((int)(sdf * 255), 255);
    if (!neg_only && pos)
      plyFile << v[0] << " " << v[1] << " " << v[2] << " " << 0 << " " << 0 << " " << sdf_i << "\n";
    if (!pos_only && neg)
      plyFile << v[0] << " " << v[1] << " " << v[2] << " " << sdf_i << " " << 0 << " " << 0 << "\n";
  }
  plyFile.close();
}

int main(int argc, char** argv) {
  std::string mesh_file = "model.obj";
  std::string output_file = "output.npz";
  bool visualize = false;
  std::string ply_path = "";
  int num_sample = 500000;
  float variance = 0.005;
  bool save_ply = true;
  bool test_mode = false;
  std::string spatial_sample_npz = "";
  
  CLI::App app;
  app.add_option("-m,--mesh", mesh_file, "input mesh file")->required();
  app.add_option("-o,--output", output_file, "output filename (npy or npz)")->required();
  app.add_flag("-v,--visualize", visualize, "visualize the mesh");
  app.add_option("--ply", ply_path, "ply point cloud filename");
  app.add_option("-s,--num_sample", num_sample, "number of samples");
  app.add_option("--var", variance, "variance for normal distribution near the surface");
  app.add_flag("--sply,--save_ply", save_ply, "save ply");
  app.add_flag("-t,--test", test_mode, "test mode flag");
  app.add_option("-n,--spatial", spatial_sample_npz, "spatial sample npz file");
  
  CLI11_PARSE(app, argc, argv);

  if (test_mode) {
    variance = 0.05;
    num_sample = 250000;
  }

  float second_variance = variance / 10;

  if (test_mode) {
    second_variance = variance / 100;
    num_sample = 250000;
  }

  std::vector<Eigen::Vector3f> xyz_surf;
  std::vector<Eigen::Vector3f> xyz_surf_normals;

  pangolin::Geometry geom = pangolin::LoadGeometry(mesh_file);
  float radius = BoundingCubeNormalization(geom, false);
  SampleFromSurface(geom, xyz_surf, (num_sample * 47) / 50);
  
  std::cout << "num_samp_near_surf: " << xyz_surf.size() << std::endl;

  std::vector<Eigen::Vector3f> vertices;
  pangolin::Image<float> verts = 
      pangolin::get<pangolin::Image<float>>(geom.buffers["geometry"].attributes["vertex"]);

  for (int i = 0; i < verts.h; i++) {
    Eigen::Vector3f v = Eigen::Map<Eigen::Vector3f>(verts.RowPtr(i));
    vertices.push_back(v);
  }

  KdVertexList kd_points(vertices);
  KdVertexListTree kdTree(3, kd_points, nanoflann::KDTreeSingleIndexAdaptorParams(10));
  kdTree.buildIndex();

  std::vector<Eigen::Vector3f> xyz;
  std::vector<float> sdfs;

  int num_votes = 11;
  int num_rand_samples = num_sample - xyz_surf.size();

  SampleSDFNearSurface(kdTree, vertices, xyz_surf, xyz_surf_normals, xyz, sdfs, num_rand_samples, variance, second_variance, radius, num_votes);

  if (save_ply) {
    writeSDFToPLY(xyz, sdfs, "sdf_samples.ply");
  }
  if (ply_path != "") {
    writeSDFToPLY(xyz, sdfs, ply_path, false, false);
  }

  writeSDFToNPZ(xyz, sdfs, output_file, true);
}
