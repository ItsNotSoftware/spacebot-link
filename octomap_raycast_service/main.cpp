/*
 * OctoMap raycast service: ZMQ REP server that answers avatar feasibility and
 * floor-distance queries from the SpaceBotLink interface. Loads a .bt OcTree
 * once at startup and replies synchronously to JSON requests.
 */

#include <octomap/AbstractOcTree.h>
#include <octomap/OcTree.h>
#include <zmq.h>

#include <boost/property_tree/json_parser.hpp>
#include <boost/property_tree/ptree.hpp>

#include <cmath>
#include <cstring>
#include <iostream>
#include <sstream>
#include <string>

struct Vec3 {
    double x = 0.0;
    double y = 0.0;
    double z = 0.0;
};

struct Quat {
    double x = 0.0;
    double y = 0.0;
    double z = 0.0;
    double w = 1.0;
};

struct Pose {
    Vec3 position;
    Quat orientation;
};

struct Options {
    std::string map_path;
    std::string endpoint = "tcp://*:5555";
    double max_range = 50.0;
};

// 3D cross product.
static Vec3 cross(const Vec3 &a, const Vec3 &b) {
    return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z,
            a.x * b.y - a.y * b.x};
}

// 3D dot product.
static double dot(const Vec3 &a, const Vec3 &b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

// Euclidean length of a vector.
static double norm(const Vec3 &v) { return std::sqrt(dot(v, v)); }

// Normalize `v`; return `fallback` when it's too small to safely normalize.
static Vec3 normalize_or_default(const Vec3 &v, const Vec3 &fallback) {
    double n = norm(v);
    if (n < 1e-9) {
        return fallback;
    }
    return {v.x / n, v.y / n, v.z / n};
}

// Rotate `v` by quaternion `q` without building a rotation matrix.
static Vec3 rotate_vector_by_quat(const Vec3 &v, const Quat &q) {
    Vec3 q_vec{q.x, q.y, q.z};
    Vec3 t = cross(q_vec, v);
    t.x *= 2.0;
    t.y *= 2.0;
    t.z *= 2.0;

    Vec3 q_w_t{t.x * q.w, t.y * q.w, t.z * q.w};
    Vec3 cross_qvec_t = cross(q_vec, t);

    return {v.x + q_w_t.x + cross_qvec_t.x, v.y + q_w_t.y + cross_qvec_t.y,
            v.z + q_w_t.z + cross_qvec_t.z};
}

// Look up a required child object by `key`; report the missing field via `err`.
static bool get_child_required(const boost::property_tree::ptree &pt,
                               const std::string &key,
                               const boost::property_tree::ptree *&out,
                               std::string *err) {
    auto child = pt.get_child_optional(key);
    if (!child) {
        if (err) {
            *err = "Missing object field: " + key;
        }
        return false;
    }
    out = &child.get();
    return true;
}

// Read a required numeric field by `key`; report the missing/invalid name via `err`.
static bool get_double_required(const boost::property_tree::ptree &pt,
                                const std::string &key, double *out,
                                std::string *err) {
    auto value = pt.get_optional<double>(key);
    if (!value) {
        if (err) {
            *err = "Missing or invalid numeric field: " + key;
        }
        return false;
    }
    *out = *value;
    return true;
}

// Parse a {"x":..,"y":..,"z":..} child node into `out`.
static bool parse_vec3(const boost::property_tree::ptree &pt,
                       const std::string &key, Vec3 *out, std::string *err) {
    const boost::property_tree::ptree *node = nullptr;
    if (!get_child_required(pt, key, node, err)) {
        return false;
    }
    if (!get_double_required(*node, "x", &out->x, err)) {
        return false;
    }
    if (!get_double_required(*node, "y", &out->y, err)) {
        return false;
    }
    if (!get_double_required(*node, "z", &out->z, err)) {
        return false;
    }
    return true;
}

// Parse a {"x":..,"y":..,"z":..,"w":..} child node into `out`.
static bool parse_quat(const boost::property_tree::ptree &pt,
                       const std::string &key, Quat *out, std::string *err) {
    const boost::property_tree::ptree *node = nullptr;
    if (!get_child_required(pt, key, node, err)) {
        return false;
    }
    if (!get_double_required(*node, "x", &out->x, err)) {
        return false;
    }
    if (!get_double_required(*node, "y", &out->y, err)) {
        return false;
    }
    if (!get_double_required(*node, "z", &out->z, err)) {
        return false;
    }
    if (!get_double_required(*node, "w", &out->w, err)) {
        return false;
    }
    return true;
}

// Parse a pose object with required `position` and `orientation` children.
static bool parse_pose(const boost::property_tree::ptree &root,
                       const std::string &key, Pose *out, std::string *err) {
    const boost::property_tree::ptree *pose = nullptr;
    if (!get_child_required(root, key, pose, err)) {
        return false;
    }
    if (!parse_vec3(*pose, "position", &out->position, err)) {
        return false;
    }
    if (!parse_quat(*pose, "orientation", &out->orientation, err)) {
        return false;
    }
    return true;
}

// Build a JSON error reply of the form {"error": "<message>"}.
static std::string make_error_response(const std::string &message) {
    boost::property_tree::ptree response;
    response.put("error", message);
    std::ostringstream out;
    boost::property_tree::write_json(out, response, false);
    return out.str();
}

/*
 * Answer one avatar_query: cast a ray from the avatar along its local -Z to
 * find the nearest "ground" voxel, ray-test from the robot toward the avatar
 * for occlusion, and check whether the avatar position lies inside an
 * occupied voxel. Returns a JSON string with ground_distance, ground_axis,
 * avatar_occluded, and avatar_in_obstacle (or an "error" field on bad input).
 */
static std::string handle_request(const std::string &request_json,
                                  octomap::OcTree &tree, double max_range) {
    boost::property_tree::ptree root;
    try {
        std::istringstream input(request_json);
        boost::property_tree::read_json(input, root);
    } catch (const std::exception &ex) {
        return make_error_response(std::string("Invalid JSON: ") + ex.what());
    }

    // Validate request shape.
    auto type_value = root.get_optional<std::string>("type");
    if (!type_value || *type_value != "avatar_query") {
        return make_error_response("Invalid or missing type field");
    }

    Pose robot_pose;
    Pose avatar_pose;
    std::string err;
    if (!parse_pose(root, "robot_pose", &robot_pose, &err)) {
        return make_error_response(err);
    }
    if (!parse_pose(root, "avatar_pose", &avatar_pose, &err)) {
        return make_error_response(err);
    }

    // Compute avatar "down" in world space and snap to best-aligned axis.
    Vec3 local_down{0.0, 0.0, -1.0};
    Vec3 down_world =
        rotate_vector_by_quat(local_down, avatar_pose.orientation);
    down_world = normalize_or_default(down_world, local_down);

    const Vec3 axes[6] = {
        {1.0, 0.0, 0.0},  {-1.0, 0.0, 0.0}, {0.0, 1.0, 0.0},
        {0.0, -1.0, 0.0}, {0.0, 0.0, 1.0},  {0.0, 0.0, -1.0},
    };
    const char *axis_names[6] = {"x", "-x", "y", "-y", "z", "-z"};

    double best_dot = -1e9;
    int best_axis = 0;
    for (int i = 0; i < 6; ++i) {
        double d = dot(down_world, axes[i]);
        if (d > best_dot) {
            best_dot = d;
            best_axis = i;
        }
    }

    // Ground raycast from the avatar along the selected axis.
    octomap::point3d ground_origin(
        avatar_pose.position.x, avatar_pose.position.y, avatar_pose.position.z);
    octomap::point3d ground_direction(axes[best_axis].x, axes[best_axis].y,
                                      axes[best_axis].z);

    octomap::point3d ground_hit;
    bool ground_hit_found = tree.castRay(ground_origin, ground_direction,
                                         ground_hit, true, max_range);

    // If no hit, return max_range as the distance (documented behavior).
    double ground_distance = max_range;
    if (ground_hit_found) {
        ground_distance = (ground_hit - ground_origin).norm();
    }

    // Occlusion check from robot to avatar.
    Vec3 to_avatar{avatar_pose.position.x - robot_pose.position.x,
                   avatar_pose.position.y - robot_pose.position.y,
                   avatar_pose.position.z - robot_pose.position.z};
    double to_avatar_dist = norm(to_avatar);

    bool avatar_occluded = false;
    if (to_avatar_dist > 1e-6) {
        Vec3 to_avatar_dir = {to_avatar.x / to_avatar_dist,
                              to_avatar.y / to_avatar_dist,
                              to_avatar.z / to_avatar_dist};
        octomap::point3d occlusion_origin(robot_pose.position.x,
                                          robot_pose.position.y,
                                          robot_pose.position.z);
        octomap::point3d occlusion_dir(to_avatar_dir.x, to_avatar_dir.y,
                                       to_avatar_dir.z);
        octomap::point3d occlusion_hit;
        bool occlusion_hit_found =
            tree.castRay(occlusion_origin, occlusion_dir, occlusion_hit, true,
                         to_avatar_dist);
        if (occlusion_hit_found) {
            double hit_dist = (occlusion_hit - occlusion_origin).norm();
            if (hit_dist + 1e-6 < to_avatar_dist) {
                avatar_occluded = true;
            }
        }
    }

    // Check whether the avatar is inside an occupied voxel.
    bool avatar_in_obstacle = false;
    if (auto node = tree.search(avatar_pose.position.x, avatar_pose.position.y,
                                avatar_pose.position.z)) {
        if (tree.isNodeOccupied(node)) {
            avatar_in_obstacle = true;
        }
    }

    // Assemble JSON response.
    boost::property_tree::ptree response;
    response.put("ground_distance", ground_distance);
    response.put("ground_axis", axis_names[best_axis]);
    response.put("avatar_occluded", avatar_occluded);
    response.put("avatar_in_obstacle", avatar_in_obstacle);

    std::ostringstream out;
    boost::property_tree::write_json(out, response, false);
    return out.str();
}

// Print CLI usage to stderr.
static void print_usage(const char *program) {
    std::cerr << "Usage: " << program << " <map.bt> [endpoint] [max_range]\n";
}

// Parse positional CLI args into `options`; returns false on missing/invalid input.
static bool parse_options(int argc, char **argv, Options *options) {
    if (argc < 2) {
        return false;
    }
    options->map_path = argv[1];
    if (argc >= 3) {
        options->endpoint = argv[2];
    }
    if (argc >= 4) {
        try {
            options->max_range = std::stod(argv[3]);
        } catch (const std::exception &) {
            return false;
        }
    }
    return true;
}

// Entry point: load the map, bind a ZMQ REP socket, and serve queries forever.
int main(int argc, char **argv) {
    Options options;
    if (!parse_options(argc, argv, &options)) {
        print_usage(argv[0]);
        return 1;
    }

    // Load the map once at startup.
    std::unique_ptr<octomap::OcTree> tree;
    {
        octomap::AbstractOcTree *abstract_tree =
            octomap::AbstractOcTree::read(options.map_path);
        if (abstract_tree) {
            tree.reset(dynamic_cast<octomap::OcTree *>(abstract_tree));
            if (!tree) {
                delete abstract_tree;
                std::cerr << "Map is not an OcTree: " << options.map_path
                          << "\n";
                return 1;
            }
        } else {
            // Fallback for binary .bt headers not handled by
            // AbstractOcTree::read.
            auto fallback_tree = std::make_unique<octomap::OcTree>(0.1);
            if (!fallback_tree->readBinary(options.map_path)) {
                std::cerr << "Failed to read OctoMap: " << options.map_path
                          << "\n";
                return 1;
            }
            tree = std::move(fallback_tree);
        }
    }

    // Set up ZMQ REP socket.
    void *context = zmq_ctx_new();
    if (!context) {
        std::cerr << "Failed to create ZMQ context\n";
        return 1;
    }

    void *socket = zmq_socket(context, ZMQ_REP);
    if (!socket) {
        std::cerr << "Failed to create ZMQ socket\n";
        zmq_ctx_term(context);
        return 1;
    }

    if (zmq_bind(socket, options.endpoint.c_str()) != 0) {
        std::cerr << "Failed to bind ZMQ endpoint: " << options.endpoint << " ("
                  << zmq_strerror(zmq_errno()) << ")\n";
        zmq_close(socket);
        zmq_ctx_term(context);
        return 1;
    }

    std::cout << "OctoMap raycast service listening on " << options.endpoint
              << "\n";

    while (true) {
        // Receive a request, process, and reply synchronously.
        zmq_msg_t request_msg;
        zmq_msg_init(&request_msg);
        int recv_rc = zmq_msg_recv(&request_msg, socket, 0);
        if (recv_rc < 0) {
            zmq_msg_close(&request_msg);
            continue;
        }
        const char *request_data =
            static_cast<const char *>(zmq_msg_data(&request_msg));
        size_t request_size = zmq_msg_size(&request_msg);
        std::string request_json(request_data, request_size);
        zmq_msg_close(&request_msg);

        std::string response_json =
            handle_request(request_json, *tree, options.max_range);

        zmq_msg_t reply_msg;
        zmq_msg_init_size(&reply_msg, response_json.size());
        std::memcpy(zmq_msg_data(&reply_msg), response_json.data(),
                    response_json.size());
        int send_rc = zmq_msg_send(&reply_msg, socket, 0);
        zmq_msg_close(&reply_msg);
        if (send_rc < 0) {
            continue;
        }
    }

    zmq_close(socket);
    zmq_ctx_term(context);
    return 0;
}
