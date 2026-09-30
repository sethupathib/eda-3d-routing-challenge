// M3D delay router.
//
// Objective: sum, over every sink, of the driver-to-sink path delay.
// A shortest-path tree from the driver is optimal for one net. Vertices and
// edges are exclusive across nets, so congestion is only a feasibility
// constraint. Middle layers are cheapest and oversubscribed; the search is
// PathFinder (capped, so nets spill to the next layer instead of snaking)
// followed by residual polish and large-neighborhood re-routing.
//
// Build:  make -C cpp
// Run:    cpp/m3d_route --suite benchmarks_hard --out submissions/hard/cpp --seconds 30 --jobs 4

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <fstream>
#include <functional>
#include <iostream>
#include <mutex>
#include <queue>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

using Clock = std::chrono::steady_clock;

static std::mutex g_print_mu;

// ---------------------------------------------------------------------------
// Minimal JSON DOM. Enough for the instance and suite files.
// ---------------------------------------------------------------------------

struct Json {
    enum Type { NUL, BOOL, NUM, STR, ARR, OBJ };
    Type type = NUL;
    bool b = false;
    long long n = 0;
    std::string s;
    std::vector<Json> a;
    std::vector<std::pair<std::string, Json>> o;

    const Json* get(const std::string& k) const {
        for (auto& kv : o)
            if (kv.first == k) return &kv.second;
        return nullptr;
    }
    const Json& at(const std::string& k) const {
        const Json* p = get(k);
        if (!p) throw std::runtime_error("missing json key " + k);
        return *p;
    }
};

class Parser {
public:
    explicit Parser(std::string in) : s(std::move(in)) {}
    Json parse() {
        skip();
        Json v = value();
        skip();
        return v;
    }

private:
    std::string s;
    size_t i = 0;

    void skip() {
        while (i < s.size() && (s[i] == ' ' || s[i] == '\n' || s[i] == '\r' || s[i] == '\t'))
            ++i;
    }
    char peek() {
        skip();
        return i < s.size() ? s[i] : 0;
    }
    char getc() {
        skip();
        if (i >= s.size()) throw std::runtime_error("unexpected end of json");
        return s[i++];
    }
    Json value() {
        char c = peek();
        if (c == '{') return object();
        if (c == '[') return array();
        if (c == '"') {
            Json j;
            j.type = Json::STR;
            j.s = str();
            return j;
        }
        if (c == 't' || c == 'f') return boolean();
        if (c == 'n') return null();
        return number();
    }
    Json object() {
        Json j;
        j.type = Json::OBJ;
        getc();  // {
        if (peek() == '}') {
            getc();
            return j;
        }
        while (true) {
            if (peek() != '"') throw std::runtime_error("expected string key");
            std::string k = str();
            if (getc() != ':') throw std::runtime_error("expected colon");
            j.o.emplace_back(std::move(k), value());
            char c = getc();
            if (c == '}') break;
            if (c != ',') throw std::runtime_error("expected comma in object");
        }
        return j;
    }
    Json array() {
        Json j;
        j.type = Json::ARR;
        getc();
        if (peek() == ']') {
            getc();
            return j;
        }
        while (true) {
            j.a.push_back(value());
            char c = getc();
            if (c == ']') break;
            if (c != ',') throw std::runtime_error("expected comma in array");
        }
        return j;
    }
    std::string str() {
        if (getc() != '"') throw std::runtime_error("expected string");
        std::string out;
        while (i < s.size()) {
            char c = s[i++];
            if (c == '"') return out;
            if (c == '\\') {
                if (i >= s.size()) throw std::runtime_error("bad escape");
                char e = s[i++];
                if (e == 'n') out.push_back('\n');
                else if (e == 't') out.push_back('\t');
                else if (e == 'r') out.push_back('\r');
                else out.push_back(e);
            } else {
                out.push_back(c);
            }
        }
        throw std::runtime_error("unterminated string");
    }
    Json boolean() {
        Json j;
        j.type = Json::BOOL;
        if (s.compare(i, 4, "true") == 0) {
            i += 4;
            j.b = true;
        } else if (s.compare(i, 5, "false") == 0) {
            i += 5;
            j.b = false;
        } else {
            throw std::runtime_error("bad bool");
        }
        return j;
    }
    Json null() {
        if (s.compare(i, 4, "null") != 0) throw std::runtime_error("bad null");
        i += 4;
        Json j;
        j.type = Json::NUL;
        return j;
    }
    Json number() {
        skip();
        size_t start = i;
        if (i < s.size() && (s[i] == '-' || s[i] == '+')) ++i;
        while (i < s.size() && s[i] >= '0' && s[i] <= '9') ++i;
        if (i < s.size() && s[i] == '.') {
            ++i;
            while (i < s.size() && s[i] >= '0' && s[i] <= '9') ++i;
        }
        if (start == i) throw std::runtime_error("bad number");
        Json j;
        j.type = Json::NUM;
        j.n = std::stoll(s.substr(start, i - start));
        return j;
    }
};

static Json load_json(const std::string& path) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error("cannot open " + path);
    std::ostringstream ss;
    ss << in.rdbuf();
    return Parser(ss.str()).parse();
}

// ---------------------------------------------------------------------------
// Instance
// ---------------------------------------------------------------------------

struct Net {
    int id = 0;
    int driver = 0;             // vertex id
    std::vector<int> sinks;     // vertex ids
    int lb = 0;
    int x0 = 0, y0 = 0, x1 = 0, y1 = 0;
};

struct Route {
    std::vector<int> verts;
    std::vector<int> eids;
    int delay = 0;
    bool ok = false;
};

struct Instance {
    std::string name;
    int W = 0, H = 0, L = 0;
    std::vector<int> layer_delay;
    int via = 0;
    std::vector<Net> nets;
    std::vector<int> pin_owner;  // vertex -> net slot, or -1
};

static Instance load_instance(const std::string& path) {
    Json root = load_json(path);
    Instance inst;
    inst.name = root.at("name").s;
    const Json& grid = root.at("grid");
    inst.W = (int)grid.at("width").n;
    inst.H = (int)grid.at("height").n;
    inst.L = (int)grid.at("layers").n;
    const Json& delay = root.at("delay");
    for (const Json& x : delay.at("layer_delay").a) inst.layer_delay.push_back((int)x.n);
    if ((int)inst.layer_delay.size() != inst.L)
        throw std::runtime_error("layer_delay length mismatch in " + path);
    inst.via = (int)delay.at("via_delay").n;

    const int WH = inst.W * inst.H;
    const int N = WH * inst.L;
    auto vid = [&](int x, int y, int z) { return (z * inst.H + y) * inst.W + x; };

    int max_pin = 0;
    for (const Json& p : root.at("pins").a) max_pin = std::max(max_pin, (int)p.at("id").n);
    std::vector<int> pin_vid(max_pin + 1, -1);
    std::vector<int> pin_net_slot(max_pin + 1, -1);
    for (const Json& p : root.at("pins").a) {
        int id = (int)p.at("id").n;
        pin_vid[id] = vid((int)p.at("x").n, (int)p.at("y").n, (int)p.at("z").n);
    }

    const Json& nets = root.at("nets");
    inst.nets.resize(nets.a.size());
    for (size_t si = 0; si < nets.a.size(); ++si) {
        const Json& nj = nets.a[si];
        Net& net = inst.nets[si];
        net.id = (int)nj.at("id").n;
        int driver_pin = (int)nj.at("driver").n;
        net.driver = pin_vid[driver_pin];
        pin_net_slot[driver_pin] = (int)si;
        int x, y, z;
        auto coord = [&](int v) {
            z = v / WH;
            int r = v % WH;
            y = r / inst.W;
            x = r % inst.W;
        };
        coord(net.driver);
        net.x0 = net.x1 = x;
        net.y0 = net.y1 = y;
        for (const Json& s : nj.at("sinks").a) {
            int pid = (int)s.n;
            int sv = pin_vid[pid];
            net.sinks.push_back(sv);
            pin_net_slot[pid] = (int)si;
            coord(sv);
            net.x0 = std::min(net.x0, x);
            net.y0 = std::min(net.y0, y);
            net.x1 = std::max(net.x1, x);
            net.y1 = std::max(net.y1, y);
        }
        if (net.driver < 0 || net.sinks.empty())
            throw std::runtime_error("bad net in " + path);
    }

    inst.pin_owner.assign(N, -1);
    for (int pid = 0; pid <= max_pin; ++pid) {
        if (pin_vid[pid] < 0) continue;
        int sl = pin_net_slot[pid];
        if (sl < 0) continue;
        int v = pin_vid[pid];
        if (inst.pin_owner[v] != -1 && inst.pin_owner[v] != sl)
            throw std::runtime_error("two nets share a pin vertex");
        inst.pin_owner[v] = sl;
    }
    return inst;
}

// ---------------------------------------------------------------------------
// Router
// ---------------------------------------------------------------------------

struct Adj {
    int to, w, eid;
};

class Router {
public:
    explicit Router(Instance inst) : inst(std::move(inst)) {
        W = this->inst.W;
        H = this->inst.H;
        L = this->inst.L;
        WH = W * H;
        NV = WH * L;
        build_graph();
        const int S = (int)this->inst.nets.size();
        routes.assign(S, Route{});
        owner.assign(NV, -1);
        stamp.assign(NV, 0);
        closed.assign(NV, 0);
        dist.assign(NV, 0);
        parent.assign(NV, -1);
        pdelay.assign(NV, 0);
        peid.assign(NV, -1);
        sink_mark.assign(NV, 0);
        node_delay.assign(NV, -1);
        seen_v.assign(NV, 0);
        seen_e.assign(1, 0);
        pin_owner = this->inst.pin_owner;
        compute_lb();
    }

    int nnets() const { return (int)inst.nets.size(); }
    int total() const {
        int s = 0;
        for (auto& r : routes)
            if (r.ok) s += r.delay;
        return s;
    }
    int lb_sum() const {
        int s = 0;
        for (auto& n : inst.nets) s += n.lb;
        return s;
    }
    const Instance& instance() const { return inst; }

    struct Snap {
        std::vector<Route> routes;
    };
    Snap snapshot() const { return Snap{routes}; }
    void restore(const Snap& sn) {
        clear();
        for (int i = 0; i < nnets(); ++i)
            if (sn.routes[i].ok) commit(i, sn.routes[i]);
    }

    // mode 0 hard delay, 1 penalty, 2 additive congestion, 3 multiplicative
    Route route(int si, int mode, double pres = 0, double penalty = 0,
                const std::vector<double>* h = nullptr, const std::vector<int>* occ = nullptr,
                double cap = 0) {
        ++tick;
        if (tick >= 2000000000) reset_stamps();
        const int t = tick;
        const Net& net = inst.nets[si];
        const int driver = net.driver;
        ++sink_tick;
        int need = 0;
        for (int s : net.sinks) {
            if (s == driver) continue;
            if (sink_mark[s] != sink_tick) {
                sink_mark[s] = sink_tick;
                ++need;
            }
        }
        stamp[driver] = t;
        dist[driver] = 0;
        parent[driver] = -1;
        using Item = std::pair<double, int>;
        std::priority_queue<Item, std::vector<Item>, std::greater<Item>> heap;
        heap.push({0.0, driver});
        int got = 0;
        while (!heap.empty() && got < need) {
            auto [d, u] = heap.top();
            heap.pop();
            if (closed[u] == t) continue;
            if (stamp[u] != t || d > dist[u]) continue;
            closed[u] = t;
            if (sink_mark[u] == sink_tick && u != driver) {
                ++got;
                if (got == need) break;
            }
            const double du = dist[u];
            for (const Adj& e : adj[u]) {
                const int v = e.to;
                if (pin_owner[v] != -1 && pin_owner[v] != si) continue;
                double nd;
                if (mode == 0) {
                    if (owner[v] != -1) continue;
                    nd = du + e.w;
                } else if (mode == 1) {
                    double extra = owner[v] != -1 ? penalty : 0;
                    nd = du + e.w + extra;
                } else if (mode == 2) {
                    if (owner[v] != -1) continue;
                    double add = 0;
                    if (occ) add += pres * (*occ)[v];
                    if (h) add += (*h)[v];
                    if (cap > 0) add = std::min(add, cap);
                    nd = du + e.w + add;
                } else {
                    if (owner[v] != -1) continue;
                    double factor = 1.0;
                    if (h) factor += (*h)[v];
                    if (occ) factor += pres * (*occ)[v];
                    nd = du + e.w * factor;
                }
                if (stamp[v] != t || nd < dist[v]) {
                    stamp[v] = t;
                    dist[v] = nd;
                    parent[v] = u;
                    pdelay[v] = e.w;
                    peid[v] = e.eid;
                    heap.push({nd, v});
                }
            }
        }
        if (got < need) return Route{};
        return materialize(si, t);
    }

    void clear() {
        std::fill(owner.begin(), owner.end(), -1);
        for (auto& r : routes) r = Route{};
    }

    void commit(int si, const Route& r) {
        for (int v : r.verts) {
            if (owner[v] != -1)
                throw std::runtime_error("commit conflict net " + std::to_string(inst.nets[si].id));
            owner[v] = si;
        }
        routes[si] = r;
    }

    void remove(int si) {
        for (int v : routes[si].verts) {
            if (owner[v] != si) throw std::runtime_error("remove mismatch");
            owner[v] = -1;
        }
        routes[si] = Route{};
    }

    bool reroute_hard(const std::vector<int>& nids) {
        std::vector<int> placed;
        placed.reserve(nids.size());
        for (int si : nids) {
            Route r = route(si, 0);
            if (!r.ok) {
                for (int p : placed) remove(p);
                return false;
            }
            commit(si, r);
            placed.push_back(si);
        }
        return true;
    }

    bool reroute_soft(const std::vector<int>& nids, int mode, int max_iters, double pres0,
                      double pres_mult, double hist, double cap, bool rescue) {
        if (nids.empty()) return true;
        std::vector<int> occ(NV, 0);
        std::vector<double> h(NV, 0);
        std::vector<Route> placed(nnets());
        std::vector<char> in(nnets(), 0);
        for (int si : nids) in[si] = 1;
        double pres = pres0;
        for (int si : nids) {
            Route r = route(si, mode, pres, 0, &h, &occ, cap);
            if (!r.ok) return false;
            for (int v : r.verts) occ[v] += 1;
            placed[si] = std::move(r);
        }
        for (int it = 0; it < max_iters; ++it) {
            std::vector<int> affected;
            bool any = false;
            for (int v = 0; v < NV; ++v)
                if (occ[v] > 1) any = true;
            if (!any) {
                for (int si : nids) commit(si, placed[si]);
                return true;
            }
            for (int si : nids) {
                bool bad = false;
                for (int v : placed[si].verts)
                    if (occ[v] > 1) {
                        bad = true;
                        break;
                    }
                if (bad) affected.push_back(si);
            }
            for (int v = 0; v < NV; ++v)
                if (occ[v] > 1) h[v] += hist * (occ[v] - 1);
            pres = std::min(pres * pres_mult, 1e6);
            for (int si : affected) {
                Route old = placed[si];
                for (int v : old.verts) occ[v] -= 1;
                Route r = route(si, mode, pres, 0, &h, &occ, cap);
                if (!r.ok) {
                    for (int v : old.verts) occ[v] += 1;
                    continue;
                }
                for (int v : r.verts) occ[v] += 1;
                placed[si] = std::move(r);
            }
        }
        bool any = false;
        for (int v = 0; v < NV; ++v)
            if (occ[v] > 1) any = true;
        if (!any) {
            for (int si : nids) commit(si, placed[si]);
            return true;
        }
        if (rescue && rescue_conflicts(nids, placed, occ, mode, cap)) return true;
        return false;
    }

    bool pathfinder(const std::vector<int>& order, int mode, int iters, double pres0,
                    double pres_mult, double hist, double cap) {
        clear();
        return reroute_soft(order, mode, iters, pres0, pres_mult, hist, cap, true);
    }

    int polish(int passes = 8) {
        int removed = 0;
        std::vector<int> order(nnets());
        for (int i = 0; i < nnets(); ++i) order[i] = i;
        for (int p = 0; p < passes; ++p) {
            std::sort(order.begin(), order.end(), [&](int a, int b) {
                int sa = routes[a].ok ? routes[a].delay - inst.nets[a].lb : -1;
                int sb = routes[b].ok ? routes[b].delay - inst.nets[b].lb : -1;
                if (sa != sb) return sa > sb;
                return inst.nets[a].id < inst.nets[b].id;
            });
            int gain = 0;
            for (int si : order) {
                if (!routes[si].ok) continue;
                Route old = routes[si];
                remove(si);
                Route neu = route(si, 0);
                if (neu.ok && better(neu, old)) {
                    gain += old.delay - neu.delay;
                    commit(si, neu);
                } else {
                    commit(si, old);
                }
            }
            removed += gain;
            if (gain == 0) break;
        }
        return removed;
    }

    int group_search(int rounds, int max_b) {
        const double penalties[] = {0, 1, 2, 4, 8, 16, 32, 64};
        int total_gain = 0;
        for (int rnd = 0; rnd < rounds; ++rnd) {
            std::vector<int> order(nnets());
            for (int i = 0; i < nnets(); ++i) order[i] = i;
            std::sort(order.begin(), order.end(), [&](int a, int b) {
                return (routes[a].delay - inst.nets[a].lb) > (routes[b].delay - inst.nets[b].lb);
            });
            int gain = 0;
            for (int si : order) {
                if (!routes[si].ok) continue;
                int slack = routes[si].delay - inst.nets[si].lb;
                if (slack <= 0) continue;
                Route old = routes[si];
                remove(si);
                std::vector<std::pair<Route, std::vector<int>>> cands;
                Route best_free;
                bool have_free = false;
                for (double pen : penalties) {
                    Route rec = route(si, 1, 0, pen);
                    if (!rec.ok) continue;
                    auto bl = blockers(rec, si);
                    if (bl.empty()) {
                        if (!have_free || better(rec, best_free)) {
                            best_free = std::move(rec);
                            have_free = true;
                        }
                        continue;
                    }
                    if ((int)bl.size() > max_b) continue;
                    int slack_room = std::max(40, old.delay / 4);
                    if (rec.delay > old.delay + slack_room) continue;
                    bool dup = false;
                    for (auto& c : cands)
                        if (c.second == bl) dup = true;
                    if (dup) continue;
                    cands.push_back({std::move(rec), std::move(bl)});
                }
                Route chosen = old;
                if (have_free && better(best_free, old)) chosen = best_free;
                commit(si, chosen);
                gain += old.delay - chosen.delay;
                std::sort(cands.begin(), cands.end(), [](auto& a, auto& b) {
                    if (a.first.delay != b.first.delay) return a.first.delay < b.first.delay;
                    return a.second.size() < b.second.size();
                });
                int tries = 0;
                for (auto& c : cands) {
                    if (tries >= 3) break;
                    ++tries;
                    gain += try_group(si, c.first, c.second);
                }
            }
            total_gain += gain;
            if (gain == 0) break;
        }
        return total_gain;
    }

    int pair_pass(std::mt19937& rng, int max_pairs) {
        std::vector<int> nids;
        for (int si = 0; si < nnets(); ++si)
            if (routes[si].ok && routes[si].delay > inst.nets[si].lb) nids.push_back(si);
        std::sort(nids.begin(), nids.end(), [&](int a, int b) {
            return (routes[a].delay - inst.nets[a].lb) > (routes[b].delay - inst.nets[b].lb);
        });
        std::vector<std::pair<int, int>> pairs;
        for (size_t i = 0; i < nids.size(); ++i) {
            for (size_t j = i + 1; j < nids.size(); ++j) {
                if (!overlap(nids[i], nids[j])) continue;
                pairs.emplace_back(nids[i], nids[j]);
            }
        }
        std::shuffle(pairs.begin(), pairs.end(), rng);
        if ((int)pairs.size() > max_pairs) pairs.resize(max_pairs);
        int gain = 0;
        for (auto [a, b] : pairs) {
            if (!routes[a].ok || !routes[b].ok) continue;
            Route sa = routes[a], sb = routes[b];
            int old = sa.delay + sb.delay;
            int best = old;
            Route ba, bb;
            bool have = false;
            for (int ord = 0; ord < 2; ++ord) {
                remove(a);
                remove(b);
                std::vector<int> order = ord == 0 ? std::vector<int>{a, b} : std::vector<int>{b, a};
                if (!reroute_hard(order)) {
                    commit(a, sa);
                    commit(b, sb);
                    continue;
                }
                int neu = routes[a].delay + routes[b].delay;
                if (neu < best) {
                    best = neu;
                    ba = routes[a];
                    bb = routes[b];
                    have = true;
                }
                remove(a);
                remove(b);
                commit(a, sa);
                commit(b, sb);
            }
            if (have) {
                remove(a);
                remove(b);
                commit(a, ba);
                commit(b, bb);
                gain += old - best;
            }
        }
        return gain;
    }

    int lns(int steps, std::mt19937& rng, bool spatial) {
        int gain = 0;
        std::vector<int> nids(nnets());
        for (int i = 0; i < nnets(); ++i) nids[i] = i;
        for (int step = 0; step < steps; ++step) {
            std::vector<int> weights(nnets());
            int alive = 0;
            for (int si : nids) {
                if (!routes[si].ok) continue;
                weights[si] = std::max(1, routes[si].delay - inst.nets[si].lb);
                ++alive;
            }
            if (alive < 2) break;
            int k = 3 + (int)(rng() % 8);  // 3..10
            if (step % 7 == 0) k = std::min(alive, 12 + (int)(rng() % 8));
            k = std::min(k, alive);
            std::vector<int> group;
            group.reserve(k);
            if (spatial) {
                // Seed on a high-slack net, then grow through overlapping boxes.
                int seed = weighted_pick(nids, weights, rng);
                group.push_back(seed);
                std::vector<char> used(nnets(), 0);
                used[seed] = 1;
                for (int g = 1; g < k; ++g) {
                    std::vector<int> pool;
                    std::vector<int> pw;
                    for (int si : nids) {
                        if (used[si] || !routes[si].ok) continue;
                        bool near = false;
                        for (int u : group)
                            if (overlap(u, si)) {
                                near = true;
                                break;
                            }
                        if (!near && g > 1) continue;
                        if (!near && g == 1) continue;
                        pool.push_back(si);
                        pw.push_back(weights[si]);
                    }
                    if (pool.empty()) break;
                    int pick = pool[weighted_index(pw, rng)];
                    used[pick] = 1;
                    group.push_back(pick);
                }
            } else {
                std::vector<int> pool = nids;
                std::vector<int> pw = weights;
                for (int g = 0; g < k; ++g) {
                    int idx = weighted_index(pw, rng);
                    group.push_back(pool[idx]);
                    pool.erase(pool.begin() + idx);
                    pw.erase(pw.begin() + idx);
                }
            }
            if ((int)group.size() < 2) continue;
            Snap saved = snapshot();
            int old = 0;
            for (int si : group) old += routes[si].delay;
            for (int si : group) remove(si);
            std::vector<int> order = group;
            std::sort(order.begin(), order.end(), [&](int a, int b) {
                return saved.routes[a].delay > saved.routes[b].delay;
            });
            bool ok = reroute_hard(order);
            if (!ok && step % 3 == 0) {
                ok = reroute_soft(order, 2, 18, 0.5, 1.6, 0.4, 4.0, false);
            }
            if (!ok) {
                // one shuffled hard attempt
                std::shuffle(order.begin(), order.end(), rng);
                ok = reroute_hard(order);
            }
            if (!ok) {
                restore(saved);
                continue;
            }
            int neu = 0;
            for (int si : group) neu += routes[si].delay;
            if (neu < old) {
                gain += old - neu;
            } else {
                restore(saved);
            }
        }
        return gain;
    }

    // Rip up a fraction of the high-slack nets and negotiate them again.
    bool shake(std::mt19937& rng, double frac) {
        std::vector<int> order(nnets());
        for (int i = 0; i < nnets(); ++i) order[i] = i;
        std::sort(order.begin(), order.end(), [&](int a, int b) {
            return (routes[a].delay - inst.nets[a].lb) > (routes[b].delay - inst.nets[b].lb);
        });
        int k = std::max(4, (int)std::lround(frac * nnets()));
        k = std::min(k, nnets());
        std::vector<int> group(order.begin(), order.begin() + k);
        Snap saved = snapshot();
        int old = total();
        for (int si : group) remove(si);
        std::shuffle(group.begin(), group.end(), rng);
        bool ok = reroute_soft(group, 2, 30, 0.4, 1.5, 0.35, 4.0, true);
        if (!ok) ok = reroute_hard(group);
        if (!ok || total() >= old) {
            restore(saved);
            return false;
        }
        return true;
    }

    std::vector<std::vector<int>> orders(std::mt19937& rng) const {
        std::vector<int> ids(nnets());
        for (int i = 0; i < nnets(); ++i) ids[i] = i;
        auto bbox = [&](int i) {
            const Net& n = inst.nets[i];
            return (n.x1 - n.x0) + (n.y1 - n.y0);
        };
        std::vector<int> by_bbox = ids, by_lb = ids, by_id = ids, by_asc = ids, rnd = ids;
        std::sort(by_bbox.begin(), by_bbox.end(), [&](int a, int b) {
            int da = bbox(a), db = bbox(b);
            if (da != db) return da > db;
            return inst.nets[a].id < inst.nets[b].id;
        });
        std::sort(by_asc.begin(), by_asc.end(), [&](int a, int b) {
            int da = bbox(a), db = bbox(b);
            if (da != db) return da < db;
            return inst.nets[a].id < inst.nets[b].id;
        });
        std::sort(by_lb.begin(), by_lb.end(), [&](int a, int b) {
            if (inst.nets[a].lb != inst.nets[b].lb) return inst.nets[a].lb > inst.nets[b].lb;
            return inst.nets[a].id < inst.nets[b].id;
        });
        std::sort(by_id.begin(), by_id.end(), [&](int a, int b) {
            return inst.nets[a].id < inst.nets[b].id;
        });
        std::shuffle(rnd.begin(), rnd.end(), rng);
        return {by_bbox, by_lb, by_id, by_asc, rnd};
    }

    void write(const std::string& path) const {
        std::ofstream out(path);
        if (!out) throw std::runtime_error("cannot write " + path);
        out << "{\n \"format\": \"m3d-submission\",\n \"version\": 1,\n \"instance\": \""
            << inst.name << "\",\n \"routes\": [\n";
        for (int si = 0; si < nnets(); ++si) {
            const Route& r = routes[si];
            if (!r.ok) throw std::runtime_error("missing route");
            out << "  {\"net\": " << inst.nets[si].id << ", \"edges\": [";
            for (size_t ei = 0; ei < r.eids.size(); ++ei) {
                int eid = r.eids[ei];
                int x1, y1, z1, x2, y2, z2;
                coord(eu[eid], x1, y1, z1);
                coord(ev[eid], x2, y2, z2);
                if (ei) out << ", ";
                out << "[[" << x1 << "," << y1 << "," << z1 << "],[" << x2 << "," << y2 << ","
                    << z2 << "]]";
            }
            out << "]}";
            if (si + 1 != nnets()) out << ",";
            out << "\n";
        }
        out << " ]\n}\n";
    }

    // Returns an empty string when the routing is a legal tree packing.
    std::string self_check() const {
        std::vector<int> seen(NV, -1);
        for (int si = 0; si < nnets(); ++si) {
            const Route& r = routes[si];
            if (!r.ok) return "missing route";
            if (r.verts.empty()) return "empty route";
            if ((int)r.eids.size() != (int)r.verts.size() - 1) return "not a tree size";
            for (int v : r.verts) {
                if (v < 0 || v >= NV) return "vertex out of range";
                if (pin_owner[v] != -1 && pin_owner[v] != si) return "through foreign pin";
                if (seen[v] != -1) return "vertex short";
                seen[v] = si;
            }
            std::vector<std::vector<std::pair<int, int>>> g(NV);
            for (int eid : r.eids) {
                int a = eu[eid], b = ev[eid];
                g[a].push_back({b, ed[eid]});
                g[b].push_back({a, ed[eid]});
            }
            std::vector<int> distv(NV, -1);
            distv[inst.nets[si].driver] = 0;
            std::vector<int> st = {inst.nets[si].driver};
            while (!st.empty()) {
                int u = st.back();
                st.pop_back();
                for (auto [v, w] : g[u]) {
                    if (distv[v] < 0) {
                        distv[v] = distv[u] + w;
                        st.push_back(v);
                    }
                }
            }
            int delay = 0;
            for (int s : inst.nets[si].sinks) {
                if (distv[s] < 0) return "disconnected sink";
                delay += distv[s];
            }
            if (delay != r.delay) return "delay mismatch";
        }
        return {};
    }

private:
    Instance inst;
    int W = 0, H = 0, L = 0, WH = 0, NV = 0;
    std::vector<Adj> adj_storage;
    std::vector<std::vector<Adj>> adj;
    std::vector<int> eu, ev, ed;
    std::vector<int> owner;
    std::vector<int> pin_owner;
    std::vector<Route> routes;

    int tick = 0;
    int sink_tick = 0;
    std::vector<int> stamp, closed, parent, pdelay, peid, sink_mark, node_delay, seen_v, seen_e;
    std::vector<double> dist;

    void reset_stamps() {
        std::fill(stamp.begin(), stamp.end(), 0);
        std::fill(closed.begin(), closed.end(), 0);
        std::fill(sink_mark.begin(), sink_mark.end(), 0);
        std::fill(seen_v.begin(), seen_v.end(), 0);
        tick = 1;
        sink_tick = 1;
    }

    void coord(int v, int& x, int& y, int& z) const {
        z = v / WH;
        int r = v % WH;
        y = r / W;
        x = r % W;
    }

    void link(int u, int v, int w) {
        int eid = (int)eu.size();
        eu.push_back(u);
        ev.push_back(v);
        ed.push_back(w);
        adj[u].push_back({v, w, eid});
        adj[v].push_back({u, w, eid});
    }

    void build_graph() {
        adj.assign(NV, {});
        for (int z = 0; z < L; ++z) {
            int ld = inst.layer_delay[z];
            for (int y = 0; y < H; ++y) {
                for (int x = 0; x < W; ++x) {
                    int u = (z * H + y) * W + x;
                    if (x + 1 < W) link(u, u + 1, ld);
                    if (y + 1 < H) link(u, u + W, ld);
                    if (z + 1 < L) link(u, u + WH, inst.via);
                }
            }
        }
        seen_e.assign(eu.size(), 0);
    }

    void compute_lb() {
        for (int si = 0; si < nnets(); ++si) {
            Route r = route(si, 0);
            if (!r.ok) throw std::runtime_error("net disconnected with an empty grid");
            inst.nets[si].lb = r.delay;
        }
    }

    Route materialize(int si, int t) {
        const Net& net = inst.nets[si];
        ++tick;
        // node_delay / seen use a fresh generation separate from search tick.
        // Reuse sink_tick-style marks via seen_v and node_delay with `gen`.
        const int gen = tick;
        node_delay[net.driver] = 0;
        seen_v[net.driver] = gen;
        std::vector<int> verts;
        verts.push_back(net.driver);
        std::vector<int> eids;
        for (int s : net.sinks) {
            if (seen_v[s] == gen && node_delay[s] >= 0 && s != net.driver) {
                // already reached in this tree
            }
            std::vector<int> chain;
            int cur = s;
            int guard = 0;
            while (seen_v[cur] != gen) {
                if (stamp[cur] != t || parent[cur] < 0) return Route{};
                chain.push_back(cur);
                cur = parent[cur];
                if (++guard > NV) return Route{};
            }
            int acc = node_delay[cur];
            for (int k = (int)chain.size() - 1; k >= 0; --k) {
                int node = chain[k];
                acc += pdelay[node];
                node_delay[node] = acc;
                if (seen_v[node] != gen) {
                    seen_v[node] = gen;
                    verts.push_back(node);
                }
            }
            cur = s;
            guard = 0;
            while (cur != net.driver) {
                int eid = peid[cur];
                if (eid < 0) return Route{};
                if (seen_e[eid] != gen) {
                    seen_e[eid] = gen;
                    eids.push_back(eid);
                }
                cur = parent[cur];
                if (++guard > NV) return Route{};
            }
        }
        if ((int)eids.size() != (int)verts.size() - 1) return Route{};
        int delay = 0;
        for (int s : net.sinks) delay += node_delay[s];
        Route r;
        r.verts = std::move(verts);
        r.eids = std::move(eids);
        r.delay = delay;
        r.ok = true;
        return r;
    }

    static bool better(const Route& a, const Route& b) {
        if (a.delay != b.delay) return a.delay < b.delay;
        return a.verts.size() < b.verts.size();
    }

    std::vector<int> blockers(const Route& rec, int self) const {
        std::vector<int> found;
        std::vector<char> hit(nnets(), 0);
        for (int v : rec.verts) {
            int o = owner[v];
            if (o != -1 && o != self && !hit[o]) {
                hit[o] = 1;
                found.push_back(o);
            }
        }
        std::sort(found.begin(), found.end());
        return found;
    }

    int try_group(int si, const Route& rec, const std::vector<int>& bl) {
        if (bl.empty()) return 0;
        std::vector<int> group;
        group.push_back(si);
        group.insert(group.end(), bl.begin(), bl.end());
        Snap saved = snapshot();
        int old_sum = 0;
        for (int g : group) old_sum += routes[g].delay;
        for (int g : group) remove(g);
        for (int v : rec.verts) {
            if (owner[v] != -1) {
                restore(saved);
                return 0;
            }
        }
        commit(si, rec);
        std::vector<int> order = bl;
        std::sort(order.begin(), order.end(), [&](int a, int b) {
            return saved.routes[a].delay > saved.routes[b].delay;
        });
        bool ok = reroute_hard(order);
        if (!ok) ok = reroute_soft(order, 2, 22, 0.8, 1.8, 0.5, 0.0, false);
        if (!ok) {
            restore(saved);
            return 0;
        }
        int neu = 0;
        for (int g : group) neu += routes[g].delay;
        if (neu < old_sum) return old_sum - neu;
        restore(saved);
        return 0;
    }

    bool overlap(int a, int b) const {
        const Net& A = inst.nets[a];
        const Net& B = inst.nets[b];
        if (A.x1 + 2 < B.x0 || B.x1 + 2 < A.x0) return false;
        if (A.y1 + 2 < B.y0 || B.y1 + 2 < A.y0) return false;
        return true;
    }

    int weighted_pick(const std::vector<int>& ids, const std::vector<int>& weights, std::mt19937& rng) {
        std::vector<int> pw;
        pw.reserve(ids.size());
        for (int id : ids) pw.push_back(weights[id]);
        return ids[weighted_index(pw, rng)];
    }

    static int weighted_index(const std::vector<int>& w, std::mt19937& rng) {
        long long sum = 0;
        for (int x : w) sum += x;
        if (sum <= 0) return 0;
        std::uniform_int_distribution<long long> dist(0, sum - 1);
        long long r = dist(rng);
        long long acc = 0;
        for (int i = 0; i < (int)w.size(); ++i) {
            acc += w[i];
            if (acc > r) return i;
        }
        return (int)w.size() - 1;
    }

    bool rescue_conflicts(const std::vector<int>& nids, std::vector<Route>& placed,
                          std::vector<int>& occ, int mode, double cap) {
        std::vector<int> conflicts;
        for (int v = 0; v < NV; ++v)
            if (occ[v] > 1) conflicts.push_back(v);
        if (conflicts.empty() || conflicts.size() > 40) return false;
        for (int rad : {1, 2, 4}) {
            std::vector<char> zone(NV, 0);
            for (int v : conflicts) {
                int x, y, z;
                coord(v, x, y, z);
                int x0 = std::max(0, x - rad), x1 = std::min(W - 1, x + rad);
                int y0 = std::max(0, y - rad), y1 = std::min(H - 1, y + rad);
                for (int zz = 0; zz < L; ++zz)
                    for (int yy = y0; yy <= y1; ++yy)
                        for (int xx = x0; xx <= x1; ++xx)
                            zone[(zz * H + yy) * W + xx] = 1;
            }
            std::vector<int> rip, keep;
            for (int si : nids) {
                bool hit = false;
                for (int v : placed[si].verts)
                    if (zone[v]) {
                        hit = true;
                        break;
                    }
                if (hit) rip.push_back(si);
                else keep.push_back(si);
            }
            if (rip.size() < 2 || rip.size() > std::max<size_t>(30, nids.size() / 2)) continue;
            std::vector<int> committed;
            bool boom = false;
            for (int si : keep) {
                try {
                    commit(si, placed[si]);
                    committed.push_back(si);
                } catch (const std::runtime_error&) {
                    boom = true;
                    break;
                }
            }
            if (boom) {
                for (int si : committed) remove(si);
                continue;
            }
            bool ok = reroute_soft(rip, mode, 24, 0.7, 1.8, 0.5, cap, false);
            if (ok) return true;
            for (int si : committed) remove(si);
        }
        return false;
    }
};

struct SolveResult {
    std::string name;
    int delay = 0;
    int lb = 0;
    double seconds = 0;
    std::string via;
};

static SolveResult solve_file(const std::string& path, const std::string& out_dir, double seconds,
                              uint32_t seed) {
    using namespace std::chrono;
    auto t0 = Clock::now();
    auto deadline = t0 + duration_cast<Clock::duration>(duration<double>(seconds));
    auto left = [&]() {
        return duration<double>(deadline - Clock::now()).count();
    };

    Instance inst = load_instance(path);
    Router router(std::move(inst));
    std::mt19937 rng(seed);
    auto ords = router.orders(rng);

    struct Sched {
        int mode;
        double pres0, mult, hist, cap;
        int iters;
        const char* name;
    };
    const Sched scheds[] = {
        {2, 0.50, 1.80, 0.50, 0.0, 50, "s0"},
        {2, 0.40, 1.60, 0.40, 0.0, 50, "s1"},
        {2, 0.50, 2.20, 0.80, 0.0, 45, "s2"},
        {3, 0.40, 1.55, 0.35, 0.0, 40, "mul"},
    };

    struct Start {
        int delay;
        std::string tag;
        Router::Snap snap;
    };
    std::vector<Start> starts;
    auto try_start = [&](size_t oi, const Sched& sc) {
        if (!router.pathfinder(ords[oi], sc.mode, sc.iters, sc.pres0, sc.mult, sc.hist, sc.cap))
            return;
        router.polish(4);
        std::string tag = std::string(sc.name) + "-o" + std::to_string(oi);
        starts.push_back({router.total(), tag, router.snapshot()});
    };
    for (size_t oi = 0; oi < ords.size(); ++oi) try_start(oi, scheds[0]);
    for (size_t oi = 0; oi < ords.size(); ++oi) {
        if (left() < seconds * 0.55) break;
        for (size_t si = 1; si < sizeof(scheds) / sizeof(scheds[0]); ++si) {
            if (left() < seconds * 0.55) break;
            try_start(oi, scheds[si]);
        }
    }
    if (starts.empty()) {
        for (size_t oi = 0; oi < ords.size() && starts.empty(); ++oi) {
            if (!router.pathfinder(ords[oi], 2, 80, 0.5, 1.9, 0.6, 0.0)) continue;
            router.polish(6);
            starts.push_back(
                {router.total(), "fallback-o" + std::to_string(oi), router.snapshot()});
        }
    }

    SolveResult result;
    result.name = router.instance().name;
    result.lb = router.lb_sum();
    if (starts.empty()) throw std::runtime_error("no legal routing for " + result.name);

    std::sort(starts.begin(), starts.end(), [](const Start& a, const Start& b) {
        return a.delay < b.delay;
    });
    int best_delay = starts[0].delay;
    Router::Snap best_snap = starts[0].snap;
    std::string best_tag = starts[0].tag;
    int n_deep = 1;
    double window = std::max(80.0, starts[0].delay * 0.03);
    for (int i = 1; i < (int)starts.size() && i < 3; ++i) {
        if (starts[i].delay <= starts[0].delay + window) n_deep = i + 1;
    }

    auto improve_until = [&](const std::string& tag, std::mt19937& local, Clock::time_point sub_end) {
        int stale = 0;
        while (Clock::now() < sub_end && stale < 2) {
            int before = router.total();
            router.polish(3);
            if (Clock::now() >= sub_end) break;
            int g = router.group_search(1, 8);
            if (Clock::now() >= sub_end) break;
            int p = router.pair_pass(local, 400);
            if (Clock::now() >= sub_end) break;
            int steps = router.nnets() > 80 ? 16 : 30;
            int ln = router.lns(steps, local, true);
            if (g || p || ln) {
                std::lock_guard<std::mutex> lock(g_print_mu);
                std::cerr << "    " << tag << " +" << (g + p + ln) << " -> " << router.total()
                          << "\n";
            }
            if (router.total() < before) stale = 0;
            else {
                ++stale;
                router.shake(local, 0.2);
            }
        }
        router.polish(4);
    };

    double remain = std::max(1.0, left());
    double probe = std::min(6.0, remain / (n_deep + 2));
    for (int i = 0; i < n_deep; ++i) {
        auto sub_end = Clock::now() + duration_cast<Clock::duration>(duration<double>(probe));
        if (sub_end > deadline) sub_end = deadline;
        router.restore(starts[i].snap);
        std::mt19937 local(seed + 17 * (i + 1));
        improve_until(starts[i].tag, local, sub_end);
        if (router.total() < best_delay) {
            best_delay = router.total();
            best_snap = router.snapshot();
            best_tag = starts[i].tag;
        }
    }
    if (left() > 0.5) {
        router.restore(best_snap);
        std::mt19937 local(seed + 99);
        improve_until(best_tag, local, deadline);
        router.polish(4);
        if (router.total() < best_delay) {
            best_delay = router.total();
            best_snap = router.snapshot();
        }
    }
    router.restore(best_snap);
    router.polish(8);
    std::string err = router.self_check();
    if (!err.empty()) throw std::runtime_error(result.name + " self-check: " + err);

    std::string out_path = out_dir + "/" + result.name + ".sol.json";
    router.write(out_path);
    result.delay = router.total();
    result.via = best_tag;
    result.seconds = std::chrono::duration<double>(Clock::now() - t0).count();
    {
        std::lock_guard<std::mutex> lock(g_print_mu);
        std::cout << result.name << " delay=" << result.delay << " lb=" << result.lb
                  << " via=" << result.via << " starts=" << starts.size()
                  << " in " << result.seconds << "s\n"
                  << std::flush;
    }
    return result;
}

static void usage() {
    std::cerr << "usage: m3d_route --suite DIR --out DIR [--seconds N] [--jobs N] [--seed N]\n"
              << "       m3d_route --case FILE --out DIR [--seconds N] [--seed N]\n";
}

int main(int argc, char** argv) {
    std::string suite, out_dir;
    std::vector<std::string> cases;
    double seconds = 30;
    int jobs = 1;
    uint32_t seed = 1;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto need = [&](const char* name) {
            if (i + 1 >= argc) throw std::runtime_error(std::string("missing value for ") + name);
            return std::string(argv[++i]);
        };
        if (a == "--suite") suite = need("--suite");
        else if (a == "--case") cases.push_back(need("--case"));
        else if (a == "--out") out_dir = need("--out");
        else if (a == "--seconds") seconds = std::stod(need("--seconds"));
        else if (a == "--jobs") jobs = std::stoi(need("--jobs"));
        else if (a == "--seed") seed = (uint32_t)std::stoul(need("--seed"));
        else if (a == "--help") {
            usage();
            return 0;
        } else {
            usage();
            return 2;
        }
    }
    if (out_dir.empty() || (suite.empty() && cases.empty())) {
        usage();
        return 2;
    }
    if (!suite.empty()) {
        Json man = load_json(suite + "/suite.json");
        cases.clear();
        for (const Json& c : man.at("cases").a)
            cases.push_back(suite + "/" + c.at("instance_file").s);
    }

    std::string mkdir = "mkdir -p " + out_dir;
    if (std::system(mkdir.c_str()) != 0) {
        std::cerr << "cannot create " << out_dir << "\n";
        return 1;
    }

    jobs = std::max(1, jobs);
    std::vector<SolveResult> results(cases.size());
    std::exception_ptr eptr;
    std::mutex err_mu;
    bool any_fail = false;
    size_t next = 0;
    std::mutex qmu;
    auto worker = [&]() {
        while (true) {
            size_t idx;
            {
                std::lock_guard<std::mutex> lock(qmu);
                if (next >= cases.size()) return;
                idx = next++;
            }
            try {
                results[idx] = solve_file(cases[idx], out_dir, seconds, seed + (uint32_t)idx * 17);
            } catch (...) {
                std::lock_guard<std::mutex> lock(err_mu);
                if (!any_fail) {
                    any_fail = true;
                    eptr = std::current_exception();
                }
                return;
            }
        }
    };
    if (jobs == 1) {
        worker();
    } else {
        std::vector<std::thread> pool;
        for (int t = 0; t < jobs; ++t) pool.emplace_back(worker);
        for (auto& th : pool) th.join();
    }
    if (any_fail) std::rethrow_exception(eptr);

    std::ofstream rt(out_dir + "/runtime.json");
    rt << "{\n";
    for (size_t i = 0; i < results.size(); ++i) {
        rt << " \"" << results[i].name << "\": " << results[i].seconds;
        if (i + 1 != results.size()) rt << ",";
        rt << "\n";
    }
    rt << "}\n";
    return 0;
}
