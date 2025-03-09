#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <vector>
#include <queue>
#include <chrono>
#include <limits>
#include <cmath>
#include <algorithm>

namespace py = pybind11;

class MinimumCostFlow {
public:
    using ll = long long;
    const ll INF = 1e18;

    struct Edge {
        int to, rev;
        ll cap;
        double cost;
        py::object action;
    };

    int n;
    std::vector<std::vector<Edge>> graph;

    MinimumCostFlow(int n_) : n(n_), graph(n_) {}

    // エッジを追加する（逆辺も自動的に追加）
    void add_edge(int f, int t, ll capacity, double cost, py::object action) {
        graph[f].push_back({t, static_cast<int>(graph[t].size()), capacity, cost, action});
        // 逆辺の action は -1（または py::none でも可）
        graph[t].push_back({f, static_cast<int>(graph[f].size()) - 1, 0, -cost, py::int_(-1)});
    }

    // s から t へ f の流量を流す。タイムアウト（秒）を超えたら -2 を返す
    double flow(int s, int t, ll f, double timeout = 1.0) {
        double res = 0;
        std::vector<double> h(n, 0.0);     // ポテンシャル
        std::vector<double> dist(n, INF);
        std::vector<int> prev_v(n), prev_e(n);

        auto start_time = std::chrono::steady_clock::now();

        while (f > 0) {
            // タイムアウトチェック
            auto now = std::chrono::steady_clock::now();
            double elapsed = std::chrono::duration<double>(now - start_time).count();
            if (elapsed > timeout)
                return -2;

            // 距離を INF で初期化
            std::fill(dist.begin(), dist.end(), static_cast<double>(INF));
            dist[s] = 0;

            // (距離, 頂点) のペアを管理する最小ヒープ
            using P = std::pair<double, int>;
            std::priority_queue<P, std::vector<P>, std::greater<P>> pq;
            pq.push({0, s});

            while (!pq.empty()) {
                auto p = pq.top();
                pq.pop();
                int v = p.second;
                if (dist[v] < p.first - 1e-9)
                    continue;
                // 隣接する各エッジを緩和
                for (int i = 0; i < int(graph[v].size()); i++) {
                    Edge &e = graph[v][i];
                    if (e.cap > 0 && dist[e.to] > dist[v] + e.cost + h[v] - h[e.to] + 1e-9) {
                        dist[e.to] = dist[v] + e.cost + h[v] - h[e.to];
                        prev_v[e.to] = v;
                        prev_e[e.to] = i;
                        pq.push({dist[e.to], e.to});
                    }
                }
            }

            // t に到達できなければフロー計算失敗
            if (dist[t] == INF)
                return -1;

            // 到達可能なノードのみ h を更新
            for (int v = 0; v < n; v++) {
                if (dist[v] < INF)
                    h[v] += dist[v];
            }

            // s-t 経路上で流せる流量の最小値を d とする
            ll d = f;
            for (int v = t; v != s; v = prev_v[v]) {
                d = std::min(d, graph[prev_v[v]][prev_e[v]].cap);
            }
            f -= d;
            res += d * h[t];

            // 経路に沿ってエッジの容量を更新
            for (int v = t; v != s; v = prev_v[v]) {
                Edge &e = graph[prev_v[v]][prev_e[v]];
                e.cap -= d;
                graph[v][e.rev].cap += d;
            }
        }
        return res;
    }
};

// MinimumCostFlow クラスの中で定義した Edge 構造体のバインディング
PYBIND11_MODULE(min_cost_flow, m) {
    m.doc() = "Pybind11 module for MinimumCostFlow (minimum cost flow algorithm)";

    py::class_<MinimumCostFlow::Edge>(m, "Edge")
        .def_readonly("to", &MinimumCostFlow::Edge::to)
        .def_readonly("rev", &MinimumCostFlow::Edge::rev)
        .def_readonly("cap", &MinimumCostFlow::Edge::cap)
        .def_readonly("cost", &MinimumCostFlow::Edge::cost)
        .def_readonly("action", &MinimumCostFlow::Edge::action);

    py::class_<MinimumCostFlow>(m, "MinimumCostFlow")
        .def(py::init<int>(), py::arg("n"))
        .def("add_edge", &MinimumCostFlow::add_edge,
             py::arg("f"), py::arg("t"), py::arg("capacity"), py::arg("cost"), py::arg("action"))
        .def("flow", &MinimumCostFlow::flow,
             py::arg("s"), py::arg("t"), py::arg("flow"), py::arg("timeout") = 1.0)
        // graph (エッジのリスト) を edges というプロパティとして公開
        .def_property_readonly("edges", [](MinimumCostFlow &self) {
             return self.graph;
        });
}
