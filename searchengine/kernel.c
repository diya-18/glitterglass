// Exact top-k MaxScore over precomputed BM25 impact postings.
// Terms must be passed sorted by max impact ASCENDING.
#include <stdint.h>
#ifdef _WIN32
#define EXPORT __declspec(dllexport)
#else
#define EXPORT
#endif

EXPORT int maxscore(const int32_t *D, const float *I, const int64_t *S, const int64_t *E,
                    const float *M, int n, int k, int32_t *oid, float *osc) {
    float ub[64]; int64_t p[64];
    if (n > 64) n = 64;
    float s = 0;
    for (int i = 0; i < n; i++) { s += M[i]; ub[i] = s; p[i] = S[i]; }
    int cnt = 0, f = 0; float theta = 0;
    for (;;) {
        while (f < n && ub[f] <= theta) f++;        // low-max terms become non-essential
        if (f >= n) break;
        int32_t d = INT32_MAX;
        for (int i = f; i < n; i++) if (p[i] < E[i] && D[p[i]] < d) d = D[p[i]];
        if (d == INT32_MAX) break;
        float sc = 0;
        for (int i = f; i < n; i++) if (p[i] < E[i] && D[p[i]] == d) { sc += I[p[i]]; p[i]++; }
        for (int i = f - 1; i >= 0; i--) {
            if (sc + ub[i] <= theta) { sc = -1; break; }   // can't reach top-k: skip
            int64_t lo = p[i], hi = E[i];
            while (lo < hi) { int64_t m = (lo + hi) >> 1; if (D[m] < d) lo = m + 1; else hi = m; }
            p[i] = lo;
            if (lo < E[i] && D[lo] == d) sc += I[lo];
        }
        if (sc > 0 && (cnt < k || sc > osc[k - 1])) {
            int j = cnt < k ? cnt++ : k - 1;
            while (j > 0 && osc[j - 1] < sc) { osc[j] = osc[j - 1]; oid[j] = oid[j - 1]; j--; }
            osc[j] = sc; oid[j] = d;
            if (cnt == k) theta = osc[k - 1];
        }
    }
    return cnt;
}
