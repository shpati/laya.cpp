/*
 * laya.c -- single-file, libc-only C port of the Laya inference engine
 *           (https://github.com/NandhaKishorM/laya, Apache-2.0).
 *
 * Ports: laya/common.py (build_sequence, DecisionModel forward, calibration) and
 *        laya/agent.py  (Agent.system_one / predict), for the English checkpoint
 *        convaiinnovations/laya (ModernBERT-large encoder + decision head).
 *
 * Build:   Linux:    cc -O3 -march=native -fopenmp laya.c -o laya -lm
 *          Windows:  gcc -O3 -march=native -fopenmp -static laya.c -o laya.exe -lws2_32 -lm     (MinGW-w64 / MSYS2)
 *                    (-static avoids shipping libgomp-1.dll / libwinpthread-1.dll; MSVC's cl is not supported)
 * Run:     ./laya MODEL_DIR request.json          ("-" reads the request from stdin)
 *          ./laya MODEL_DIR --tokenize "text"     (print token ids)
 *          ./laya MODEL_DIR --list-tensors
 *          ./laya MODEL_DIR --serve [PORT] [--bind ADDR]   HTTP server (default 127.0.0.1:29417), model loaded once;
 *                    GET /status, POST /predict (same JSON as the CLI). Linux/macOS: process per connection;
 *                    Windows: thread per connection. Inference is serialized, /status answers while it runs.
 * Speed:   ALWAYS build with -march=native (the GEMM uses whatever SIMD width the compiler targets: AVX2/AVX-512/NEON).
 *          Measured on one AVX-512 core: ~0.7 s for a ~90-token question, ~4 s at the full 512 tokens (~80 GFLOP/s);
 *          scales with cores under OpenMP. Load: ~2 s, ~1.9 GB RSS. Questions run one after another.
 * Env:     LAYA_DEBUG=1  dump serialized state, token ids, raw logits, timing to stderr
 *          OMP_NUM_THREADS=N  thread count when built with OpenMP
 *
 * MODEL_DIR is a local copy of the HF repo, needing only:
 *      model.safetensors  rl_agent_config.json  tokenizer.json  config.json
 *
 * request.json = {"state": <string|object|array>, "questions": {id: {type, instructions, criteria}}}
 * Output is one line of JSON with the same shape as Agent.predict()'s result dict.
 *
 * Embedding: compile with -DLAYA_NO_MAIN and use laya_load / laya_predict / laya_free.
 *
 * KNOWN DIFFERENCES FROM THE PYTHON PATH (all in the tokenizer, none in the model):
 *   - NFC normalization is not implemented (identical for NFC input, which is almost all text).
 *   - The GPT-2 pre-tokenizer regex needs Unicode \p{L}/\p{N}/\s tables; ASCII and Latin-1 are
 *     exact, other scripts use compact range tables (see ucls()) and can split differently.
 *   - Weights are converted to float32 in RAM (~1.7 GB for the 421M-parameter checkpoint).
 *   - Numerics are float32 with a fixed summation order, so results agree with PyTorch to ~1e-4,
 *     not bit-for-bit.
 */
#ifdef __MINGW32__
#define __USE_MINGW_ANSI_STDIO 1 /* %zu etc. in the MinGW msvcrt */
#endif
#define _POSIX_C_SOURCE 200809L
#define _DEFAULT_SOURCE
#define _DARWIN_C_SOURCE
#define _FILE_OFFSET_BITS 64
#include <math.h>
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <ctype.h>
#include <errno.h>
#include <setjmp.h>
#ifdef _OPENMP
#include <omp.h>
#endif
/* Strict -std=c11 turns off FMA contraction (~25% slower GEMM); allow it explicitly. */
#if defined(__clang__)
#pragma clang fp contract(fast)
#elif defined(__GNUC__)
#pragma GCC optimize("fp-contract=fast")
#endif

#define LAYA_SERVER 1
#ifdef _WIN32
#ifndef _WIN32_WINNT
#define _WIN32_WINNT 0x0600 /* inet_pton */
#endif
#include <winsock2.h>
#include <ws2tcpip.h>
#include <windows.h>
#include <process.h>
#else
#include <arpa/inet.h>
#include <fcntl.h>
#include <netinet/in.h>
#include <signal.h>
#include <sys/mman.h>
#include <sys/socket.h>
#include <sys/time.h>
#include <sys/wait.h>
#include <unistd.h>
#ifndef MAP_ANONYMOUS
#define MAP_ANONYMOUS MAP_ANON
#endif
#endif

#ifdef _WIN32
#define FSEEK _fseeki64
#else
#define FSEEK fseeko
#endif

/* ================================================================== base */

#if defined(_MSC_VER)
#define LAYA_TLS __declspec(thread)
#else
#define LAYA_TLS _Thread_local
#endif

/* In the HTTP server each connection is handled by a forked child (POSIX) or a thread (Windows) that sets g_jmp,
 * so a failing request unwinds to an error response instead of exiting. In CLI mode g_jmp is NULL and die()
 * prints and exits. */
#if defined(__GNUC__) || defined(__clang__)
/* The compiler builtins restore registers without unwinding, which is robust across threads and on 64-bit MinGW
 * (where longjmp goes through SEH unwinding). */
typedef void *laya_jmp[5];
#define LAYA_SETJMP(b) __builtin_setjmp(b)
#define LAYA_LONGJMP(b) __builtin_longjmp((b), 1)
#else
typedef jmp_buf laya_jmp;
#define LAYA_SETJMP(b) setjmp(b)
#define LAYA_LONGJMP(b) longjmp((b), 1)
#endif
static LAYA_TLS laya_jmp *g_jmp;
static LAYA_TLS char g_errmsg[512];

static void die(const char *fmt, ...)
{
    va_list ap;
    va_start(ap, fmt);
    if (g_jmp) {
        vsnprintf(g_errmsg, sizeof g_errmsg, fmt, ap);
        va_end(ap);
        LAYA_LONGJMP(*g_jmp);
    }
    fputs("laya: ", stderr);
    vfprintf(stderr, fmt, ap);
    va_end(ap);
    fputc('\n', stderr);
    exit(1);
}

/* All heap memory goes through xmalloc/xrealloc/xfree (free() is redirected below). While request tracking is on
 * (per thread), live blocks are chained so a request that dies mid-way can be released in one sweep -- this keeps a
 * long-running server leak-free under malformed requests without any per-call cleanup code. */
typedef struct AHdr {
    struct AHdr *prev, *next;
    size_t tracked, pad; /* 32-byte header keeps 16-byte alignment */
} AHdr;
static LAYA_TLS AHdr *g_tr_head;
static LAYA_TLS int g_tr_on;

static void *xmalloc(size_t n)
{
    AHdr *h = malloc(sizeof(AHdr) + (n ? n : 1));
    if (!h) die("out of memory (%zu bytes)", n);
    h->tracked = (size_t)g_tr_on;
    h->prev = NULL;
    h->next = NULL;
    if (g_tr_on) {
        h->next = g_tr_head;
        if (g_tr_head) g_tr_head->prev = h;
        g_tr_head = h;
    }
    return h + 1;
}
static void *xrealloc(void *q, size_t n)
{
    if (!q) return xmalloc(n);
    AHdr *h = (AHdr *)q - 1, *nh = realloc(h, sizeof(AHdr) + (n ? n : 1));
    if (!nh) die("out of memory (%zu bytes)", n);
    if (nh != h && nh->tracked) {
        if (nh->prev) nh->prev->next = nh;
        else g_tr_head = nh;
        if (nh->next) nh->next->prev = nh;
    }
    return nh + 1;
}
static void xfree(void *p)
{
    if (!p) return;
    AHdr *h = (AHdr *)p - 1;
    if (h->tracked) {
        if (h->prev) h->prev->next = h->next;
        else g_tr_head = h->next;
        if (h->next) h->next->prev = h->prev;
    }
    free(h);
}
#define free xfree
/* Raw malloc/calloc/realloc/strdup must not be mixed with free() from here on. */
#define malloc   LAYA_use_xmalloc_instead
#define calloc   LAYA_use_xcalloc_instead
#define realloc  LAYA_use_xrealloc_instead
#define strdup   LAYA_use_xstrdup_instead
static void *xcalloc(size_t n, size_t m)
{
    void *p = xmalloc(n * m);
    memset(p, 0, n * m);
    return p;
}
static char *xstrdup(const char *s)
{
    size_t n = strlen(s);
    char *r = xmalloc(n + 1);
    memcpy(r, s, n + 1);
    return r;
}
static void req_begin(void) { g_tr_on = 1; }
static void req_end(void) /* success: whatever is still allocated now belongs to the caller */
{
    for (AHdr *h = g_tr_head; h;) {
        AHdr *nx = h->next;
        h->tracked = 0;
        h->prev = h->next = NULL;
        h = nx;
    }
    g_tr_head = NULL;
    g_tr_on = 0;
}
static void req_abort(void) /* failure: release everything the request still holds */
{
    g_tr_on = 0;
    for (AHdr *h = g_tr_head; h;) {
        AHdr *nx = h->next;
        free(h + 1);
        h = nx;
    }
    g_tr_head = NULL;
}

static double now_ms(void)
{
#if defined(_WIN32)
    LARGE_INTEGER f, c;
    QueryPerformanceFrequency(&f);
    QueryPerformanceCounter(&c);
    return (double)c.QuadPart * 1e3 / (double)f.QuadPart;
#elif defined(CLOCK_MONOTONIC)
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1e3 + ts.tv_nsec / 1e6;
#else
    return (double)clock() * 1e3 / CLOCKS_PER_SEC; /* coarse fallback (CPU time) */
#endif
}

static char *read_file(const char *path, size_t *len_out)
{
    FILE *f = strcmp(path, "-") == 0 ? stdin : fopen(path, "rb");
    if (!f) die("cannot open %s", path);
    size_t cap = 1 << 16, n = 0;
    char *b = xmalloc(cap + 1);
    for (;;) {
        size_t r = fread(b + n, 1, cap - n, f);
        n += r;
        if (n < cap) break;
        cap *= 2;
        b = xrealloc(b, cap + 1);
    }
    if (f != stdin) fclose(f);
    b[n] = 0;
    if (len_out) *len_out = n;
    return b;
}

/* ---- growable byte buffer ---- */
typedef struct {
    char *s;
    size_t n, cap;
} Buf;

static void buf_put(Buf *b, const char *s, size_t n)
{
    if (b->n + n + 1 > b->cap) {
        b->cap = (b->cap ? b->cap * 2 : 256);
        while (b->cap < b->n + n + 1) b->cap *= 2;
        b->s = xrealloc(b->s, b->cap);
    }
    memcpy(b->s + b->n, s, n);
    b->n += n;
    b->s[b->n] = 0;
}
static void buf_puts(Buf *b, const char *s) { buf_put(b, s, strlen(s)); }
static void buf_putc(Buf *b, char c) { buf_put(b, &c, 1); }

/* ---- growable int vector ---- */
typedef struct {
    int *a;
    int n, cap;
} IVec;
static void iv_push(IVec *v, int x)
{
    if (v->n == v->cap) {
        v->cap = v->cap ? v->cap * 2 : 64;
        v->a = xrealloc(v->a, sizeof(int) * (size_t)v->cap);
    }
    v->a[v->n++] = x;
}

/* ================================================================== arena */

typedef struct Chunk {
    struct Chunk *next;
    size_t cap, used;
} Chunk;
typedef struct {
    Chunk *head;
} Arena;

static void *arena_alloc(Arena *a, size_t n)
{
    n = (n + 15) & ~(size_t)15;
    if (!a->head || a->head->used + n > a->head->cap) {
        size_t cap = n > (1u << 20) ? n : (1u << 20);
        Chunk *c = xmalloc(sizeof(Chunk) + cap + 16);
        c->next = a->head;
        c->cap = cap;
        c->used = 0;
        a->head = c;
    }
    char *base = (char *)(((uintptr_t)(a->head + 1) + 15) & ~(uintptr_t)15);
    void *p = base + a->head->used;
    a->head->used += n;
    return p;
}
static void arena_free(Arena *a)
{
    Chunk *c = a->head;
    while (c) {
        Chunk *nx = c->next;
        free(c);
        c = nx;
    }
    a->head = NULL;
}

/* ================================================================== UTF-8 */

static uint32_t utf8_dec(const unsigned char *s, size_t i, size_t end, int *n)
{
    unsigned c = s[i];
#define CONT(k) (i + (k) < end && (s[i + (k)] & 0xC0) == 0x80)
    if (c < 0x80) {
        *n = 1;
        return c;
    }
    if (c >= 0xC2 && c <= 0xDF && CONT(1)) {
        *n = 2;
        return ((c & 0x1Fu) << 6) | (s[i + 1] & 0x3Fu);
    }
    if (c >= 0xE0 && c <= 0xEF && CONT(1) && CONT(2)) {
        uint32_t cp = ((c & 0x0Fu) << 12) | ((s[i + 1] & 0x3Fu) << 6) | (s[i + 2] & 0x3Fu);
        if (cp >= 0x800 && !(cp >= 0xD800 && cp <= 0xDFFF)) {
            *n = 3;
            return cp;
        }
    }
    if (c >= 0xF0 && c <= 0xF4 && CONT(1) && CONT(2) && CONT(3)) {
        uint32_t cp = ((c & 0x07u) << 18) | ((s[i + 1] & 0x3Fu) << 12) | ((s[i + 2] & 0x3Fu) << 6) | (s[i + 3] & 0x3Fu);
        if (cp >= 0x10000 && cp <= 0x10FFFF) {
            *n = 4;
            return cp;
        }
    }
#undef CONT
    *n = 1;
    return 0xFFFD; /* invalid byte: treated as one "other" character */
}

static int utf8_enc(uint32_t cp, char *o)
{
    if (cp < 0x80) {
        o[0] = (char)cp;
        return 1;
    }
    if (cp < 0x800) {
        o[0] = (char)(0xC0 | (cp >> 6));
        o[1] = (char)(0x80 | (cp & 0x3F));
        return 2;
    }
    if (cp < 0x10000) {
        o[0] = (char)(0xE0 | (cp >> 12));
        o[1] = (char)(0x80 | ((cp >> 6) & 0x3F));
        o[2] = (char)(0x80 | (cp & 0x3F));
        return 3;
    }
    o[0] = (char)(0xF0 | (cp >> 18));
    o[1] = (char)(0x80 | ((cp >> 12) & 0x3F));
    o[2] = (char)(0x80 | ((cp >> 6) & 0x3F));
    o[3] = (char)(0x80 | (cp & 0x3F));
    return 4;
}

/* ================================================================== JSON */

enum { J_NULL, J_FALSE, J_TRUE, J_NUM, J_STR, J_ARR, J_OBJ };
typedef struct JV JV;
struct JV {
    int t;
    double num;
    const char *s; /* J_STR: decoded bytes; J_NUM: original literal */
    size_t len;
    size_t n, cap; /* J_ARR / J_OBJ */
    JV **v;
    const char **k;
};

typedef struct {
    const char *p, *end, *src;
    Arena *A;
} JP;

static void jerr(JP *P, const char *msg) { die("JSON parse error at byte %ld: %s", (long)(P->p - P->src), msg); }

static void jws(JP *P)
{
    while (P->p < P->end && (*P->p == ' ' || *P->p == '\t' || *P->p == '\n' || *P->p == '\r')) P->p++;
}

static int hex4(JP *P, const char *s)
{
    int v = 0;
    for (int i = 0; i < 4; i++) {
        char c = s[i];
        v <<= 4;
        if (c >= '0' && c <= '9') v |= c - '0';
        else if (c >= 'a' && c <= 'f') v |= c - 'a' + 10;
        else if (c >= 'A' && c <= 'F') v |= c - 'A' + 10;
        else jerr(P, "bad \\u escape");
    }
    return v;
}

static const char *jstring(JP *P, size_t *outlen)
{
    const char *q = P->p + 1, *s = q;
    int esc = 0;
    while (s < P->end && *s != '"') {
        if (*s == '\\') {
            esc = 1;
            s++;
        }
        s++;
    }
    if (s >= P->end) jerr(P, "unterminated string");
    size_t raw = (size_t)(s - q), n = 0;
    char *out = arena_alloc(P->A, raw + 1);
    if (!esc) {
        memcpy(out, q, raw);
        n = raw;
    } else {
        for (const char *i = q; i < s;) {
            if (*i != '\\') {
                out[n++] = *i++;
                continue;
            }
            i++;
            switch (*i) {
            case '"': out[n++] = '"'; i++; break;
            case '\\': out[n++] = '\\'; i++; break;
            case '/': out[n++] = '/'; i++; break;
            case 'b': out[n++] = '\b'; i++; break;
            case 'f': out[n++] = '\f'; i++; break;
            case 'n': out[n++] = '\n'; i++; break;
            case 'r': out[n++] = '\r'; i++; break;
            case 't': out[n++] = '\t'; i++; break;
            case 'u': {
                if (i + 5 > s) jerr(P, "truncated \\u escape");
                uint32_t cp = (uint32_t)hex4(P, i + 1);
                i += 5;
                if (cp >= 0xD800 && cp <= 0xDBFF && i + 6 <= s && i[0] == '\\' && i[1] == 'u') {
                    uint32_t lo = (uint32_t)hex4(P, i + 2);
                    if (lo >= 0xDC00 && lo <= 0xDFFF) {
                        cp = 0x10000 + ((cp - 0xD800) << 10) + (lo - 0xDC00);
                        i += 6;
                    }
                }
                n += (size_t)utf8_enc(cp, out + n);
                break;
            }
            default: jerr(P, "bad escape");
            }
        }
    }
    out[n] = 0;
    P->p = s + 1;
    *outlen = n;
    return out;
}

static void jpush(Arena *A, JV *c, const char *key, JV *val)
{
    if (c->n == c->cap) {
        size_t nc = c->cap ? c->cap * 2 : 4;
        JV **nv = arena_alloc(A, nc * sizeof(JV *));
        if (c->n) memcpy(nv, c->v, c->n * sizeof(JV *));
        c->v = nv;
        if (c->t == J_OBJ) {
            const char **nk = arena_alloc(A, nc * sizeof(char *));
            if (c->n) memcpy(nk, c->k, c->n * sizeof(char *));
            c->k = nk;
        }
        c->cap = nc;
    }
    if (c->t == J_OBJ) c->k[c->n] = key;
    c->v[c->n++] = val;
}

static JV *jvalue(JP *P, int depth)
{
    if (depth > 200) jerr(P, "nesting too deep");
    jws(P);
    if (P->p >= P->end) jerr(P, "unexpected end of input");
    JV *v = arena_alloc(P->A, sizeof(JV));
    memset(v, 0, sizeof *v);
    char c = *P->p;
    if (c == '{') {
        v->t = J_OBJ;
        P->p++;
        jws(P);
        if (P->p < P->end && *P->p == '}') {
            P->p++;
            return v;
        }
        for (;;) {
            jws(P);
            if (P->p >= P->end || *P->p != '"') jerr(P, "expected object key");
            size_t kl;
            const char *k = jstring(P, &kl);
            jws(P);
            if (P->p >= P->end || *P->p != ':') jerr(P, "expected ':'");
            P->p++;
            JV *val = jvalue(P, depth + 1);
            jpush(P->A, v, k, val);
            jws(P);
            if (P->p < P->end && *P->p == ',') {
                P->p++;
                continue;
            }
            if (P->p < P->end && *P->p == '}') {
                P->p++;
                break;
            }
            jerr(P, "expected ',' or '}'");
        }
    } else if (c == '[') {
        v->t = J_ARR;
        P->p++;
        jws(P);
        if (P->p < P->end && *P->p == ']') {
            P->p++;
            return v;
        }
        for (;;) {
            JV *val = jvalue(P, depth + 1);
            jpush(P->A, v, NULL, val);
            jws(P);
            if (P->p < P->end && *P->p == ',') {
                P->p++;
                continue;
            }
            if (P->p < P->end && *P->p == ']') {
                P->p++;
                break;
            }
            jerr(P, "expected ',' or ']'");
        }
    } else if (c == '"') {
        v->t = J_STR;
        v->s = jstring(P, &v->len);
    } else if (c == 't' && P->end - P->p >= 4 && !memcmp(P->p, "true", 4)) {
        v->t = J_TRUE;
        P->p += 4;
    } else if (c == 'f' && P->end - P->p >= 5 && !memcmp(P->p, "false", 5)) {
        v->t = J_FALSE;
        P->p += 5;
    } else if (c == 'n' && P->end - P->p >= 4 && !memcmp(P->p, "null", 4)) {
        v->t = J_NULL;
        P->p += 4;
    } else if (c == '-' || (c >= '0' && c <= '9')) {
        const char *s = P->p;
        while (P->p < P->end && strchr("+-0123456789.eE", *P->p)) P->p++;
        size_t n = (size_t)(P->p - s);
        char *lit = arena_alloc(P->A, n + 1);
        memcpy(lit, s, n);
        lit[n] = 0;
        v->t = J_NUM;
        v->s = lit;
        v->len = n;
        v->num = strtod(lit, NULL);
    } else {
        jerr(P, "unexpected character");
    }
    return v;
}

static JV *jparse(Arena *A, const char *text, size_t len)
{
    JP P = {text, text + len, text, A};
    if (len >= 3 && !memcmp(text, "\xEF\xBB\xBF", 3)) P.p += 3;
    JV *v = jvalue(&P, 0);
    jws(&P);
    if (P.p != P.end) jerr(&P, "trailing data");
    return v;
}

static const JV *jget(const JV *o, const char *key)
{
    if (!o || o->t != J_OBJ) return NULL;
    for (size_t i = 0; i < o->n; i++)
        if (strcmp(o->k[i], key) == 0) return o->v[i];
    return NULL;
}

/* ---- serialization: matches Python json.dumps default separators (", " / ": ") ---- */

static void py_float(Buf *b, double x)
{
    if (isnan(x)) {
        buf_puts(b, "NaN");
        return;
    }
    if (isinf(x)) {
        buf_puts(b, x < 0 ? "-Infinity" : "Infinity");
        return;
    }
    if (x == 0) {
        buf_puts(b, signbit(x) ? "-0.0" : "0.0");
        return;
    }
    char tmp[48];
    for (int prec = 1; prec <= 17; prec++) {
        snprintf(tmp, sizeof tmp, "%.*e", prec - 1, x);
        if (strtod(tmp, NULL) == x) break;
    }
    int neg = tmp[0] == '-';
    const char *p = tmp + neg;
    char dg[32];
    int nd = 0;
    for (; *p && *p != 'e'; p++)
        if (*p != '.') dg[nd++] = *p;
    int e10 = atoi(p + 1);
    while (nd > 1 && dg[nd - 1] == '0') nd--;
    if (neg) buf_putc(b, '-');
    if (e10 >= -4 && e10 < 16) {
        if (e10 >= 0) {
            for (int i = 0; i <= e10; i++) buf_putc(b, i < nd ? dg[i] : '0');
            buf_putc(b, '.');
            if (nd > e10 + 1) buf_put(b, dg + e10 + 1, (size_t)(nd - e10 - 1));
            else buf_putc(b, '0');
        } else {
            buf_puts(b, "0.");
            for (int i = 0; i < -e10 - 1; i++) buf_putc(b, '0');
            buf_put(b, dg, (size_t)nd);
        }
    } else {
        buf_putc(b, dg[0]);
        if (nd > 1) {
            buf_putc(b, '.');
            buf_put(b, dg + 1, (size_t)(nd - 1));
        }
        char eb[16];
        snprintf(eb, sizeof eb, "e%c%02d", e10 < 0 ? '-' : '+', abs(e10));
        buf_puts(b, eb);
    }
}

static void json_str(Buf *b, const char *s, size_t n, int ascii)
{
    buf_putc(b, '"');
    for (size_t i = 0; i < n;) {
        unsigned char c = (unsigned char)s[i];
        if (c == '"') buf_puts(b, "\\\""), i++;
        else if (c == '\\') buf_puts(b, "\\\\"), i++;
        else if (c == '\n') buf_puts(b, "\\n"), i++;
        else if (c == '\r') buf_puts(b, "\\r"), i++;
        else if (c == '\t') buf_puts(b, "\\t"), i++;
        else if (c == '\b') buf_puts(b, "\\b"), i++;
        else if (c == '\f') buf_puts(b, "\\f"), i++;
        else if (c < 0x20) {
            char e[8];
            snprintf(e, sizeof e, "\\u%04x", c);
            buf_puts(b, e);
            i++;
        } else if (c < 0x80 || !ascii) {
            buf_putc(b, (char)c);
            i++;
        } else {
            int k;
            uint32_t cp = utf8_dec((const unsigned char *)s, i, n, &k);
            char e[16];
            if (cp >= 0x10000) {
                cp -= 0x10000;
                snprintf(e, sizeof e, "\\u%04x\\u%04x", 0xD800 + (cp >> 10), 0xDC00 + (cp & 0x3FF));
            } else {
                snprintf(e, sizeof e, "\\u%04x", cp);
            }
            buf_puts(b, e);
            i += (size_t)k;
        }
    }
    buf_putc(b, '"');
}

static void json_write(Buf *b, const JV *v, int ascii)
{
    switch (v->t) {
    case J_NULL: buf_puts(b, "null"); break;
    case J_TRUE: buf_puts(b, "true"); break;
    case J_FALSE: buf_puts(b, "false"); break;
    case J_NUM: {
        const char *s = v->s;
        size_t i = (*s == '-') ? 1 : 0, j = i;
        while (s[j] >= '0' && s[j] <= '9') j++;
        if (j > i && s[j] == 0) { /* integer literal: Python prints it back as an int */
            if (strcmp(s, "-0") == 0) buf_puts(b, "0");
            else buf_put(b, s, v->len);
        } else {
            py_float(b, v->num);
        }
        break;
    }
    case J_STR: json_str(b, v->s, v->len, ascii); break;
    case J_ARR:
        buf_putc(b, '[');
        for (size_t i = 0; i < v->n; i++) {
            if (i) buf_puts(b, ", ");
            json_write(b, v->v[i], ascii);
        }
        buf_putc(b, ']');
        break;
    case J_OBJ:
        buf_putc(b, '{');
        for (size_t i = 0; i < v->n; i++) {
            if (i) buf_puts(b, ", ");
            json_str(b, v->k[i], strlen(v->k[i]), ascii);
            buf_puts(b, ": ");
            json_write(b, v->v[i], ascii);
        }
        buf_putc(b, '}');
        break;
    }
}

/* ================================================================== safetensors */

typedef struct {
    char *name;
    int dtype; /* 0=F32 1=F16 2=BF16 */
    int ndim;
    long long shape[8];
    uint64_t off0, off1;
} STensor;

typedef struct {
    FILE *f;
    uint64_t base;
    STensor *t;
    int n;
} STFile;

static void st_open(STFile *st, const char *path)
{
    uint16_t one = 1;
    if (*(uint8_t *)&one != 1) die("big-endian hosts are not supported");
    st->f = fopen(path, "rb");
    if (!st->f) die("cannot open %s", path);
    unsigned char lb[8];
    if (fread(lb, 1, 8, st->f) != 8) die("%s: truncated header", path);
    uint64_t hl = 0;
    for (int i = 7; i >= 0; i--) hl = (hl << 8) | lb[i];
    if (hl > (1ull << 30)) die("%s: implausible header size", path);
    char *hdr = xmalloc((size_t)hl + 1);
    if (fread(hdr, 1, (size_t)hl, st->f) != hl) die("%s: truncated header", path);
    hdr[hl] = 0;
    st->base = 8 + hl;
    Arena A = {0};
    JV *root = jparse(&A, hdr, (size_t)hl);
    if (root->t != J_OBJ) die("%s: bad safetensors header", path);
    st->t = xmalloc(sizeof(STensor) * (root->n + 1));
    st->n = 0;
    for (size_t i = 0; i < root->n; i++) {
        if (strcmp(root->k[i], "__metadata__") == 0) continue;
        const JV *e = root->v[i], *dt = jget(e, "dtype"), *sh = jget(e, "shape"), *of = jget(e, "data_offsets");
        if (!dt || !sh || !of || of->n != 2) die("%s: bad tensor entry %s", path, root->k[i]);
        STensor *T = &st->t[st->n++];
        T->name = xstrdup(root->k[i]);
        if (!strcmp(dt->s, "F32")) T->dtype = 0;
        else if (!strcmp(dt->s, "F16")) T->dtype = 1;
        else if (!strcmp(dt->s, "BF16")) T->dtype = 2;
        else die("tensor %s has unsupported dtype %s", root->k[i], dt->s);
        T->ndim = (int)(sh->n > 8 ? 8 : sh->n);
        for (int d = 0; d < T->ndim; d++) T->shape[d] = (long long)sh->v[d]->num;
        T->off0 = (uint64_t)of->v[0]->num;
        T->off1 = (uint64_t)of->v[1]->num;
    }
    arena_free(&A);
    free(hdr);
}

static const STensor *st_find(const STFile *st, const char *name)
{
    for (int i = 0; i < st->n; i++)
        if (strcmp(st->t[i].name, name) == 0) return &st->t[i];
    return NULL;
}

static float f16_to_f32(uint16_t h)
{
    uint32_t s = (uint32_t)(h & 0x8000u) << 16, e = (h >> 10) & 0x1F, m = h & 0x3FF, o;
    if (e == 0) {
        if (m == 0) o = s;
        else { /* subnormal */
            e = 127 - 15 + 1;
            while (!(m & 0x400)) {
                m <<= 1;
                e--;
            }
            o = s | (e << 23) | ((m & 0x3FF) << 13);
        }
    } else if (e == 31) {
        o = s | 0x7F800000u | (m << 13);
    } else {
        o = s | ((e + 127 - 15) << 23) | (m << 13);
    }
    float f;
    memcpy(&f, &o, 4);
    return f;
}

static float *st_load(STFile *st, const STensor *T, size_t want_numel)
{
    size_t numel = 1;
    for (int d = 0; d < T->ndim; d++) numel *= (size_t)T->shape[d];
    if (numel != want_numel) die("tensor %s has %zu elements, expected %zu", T->name, numel, want_numel);
    static const int esz[3] = {4, 2, 2};
    if (T->off1 - T->off0 != numel * (size_t)esz[T->dtype]) die("tensor %s: inconsistent data_offsets", T->name);
    float *out = xmalloc(numel * sizeof(float));
    if (FSEEK(st->f, (long long)(st->base + T->off0), SEEK_SET) != 0) die("seek failed for %s", T->name);
    if (T->dtype == 0) {
        if (fread(out, 4, numel, st->f) != numel) die("short read on %s", T->name);
    } else {
        enum { CH = 1 << 18 };
        uint16_t *tmp = xmalloc(CH * sizeof(uint16_t));
        for (size_t done = 0; done < numel;) {
            size_t c = numel - done < CH ? numel - done : CH;
            if (fread(tmp, 2, c, st->f) != c) die("short read on %s", T->name);
            if (T->dtype == 1) {
                for (size_t i = 0; i < c; i++) out[done + i] = f16_to_f32(tmp[i]);
            } else {
                for (size_t i = 0; i < c; i++) {
                    uint32_t u = (uint32_t)tmp[i] << 16;
                    memcpy(&out[done + i], &u, 4);
                }
            }
            done += c;
        }
        free(tmp);
    }
    return out;
}

/* ================================================================== tokenizer */

typedef struct {
    const char *s;
    int len, id, lstrip, rstrip;
} AddTok;

typedef struct {
    Arena A;
    /* vocab: string -> id (open addressing) */
    const char **vk;
    int *vlen, *vid;
    size_t vcap;
    /* merges: (idA,idB) -> (rank, merged id) */
    uint64_t *mkey;
    int *mrank, *mnew;
    size_t mcap;
    int byte2id[256];
    char bch[256][4];
    int bchlen[256];
    AddTok *add[2]; /* [0] = normalized:false (matched first), [1] = normalized:true */
    int nadd[2];
    unsigned char addfirst[2][256];
    int cls, sep, mask, pad;
    const char *mask_str;
} Tok;

static uint64_t fnv(const char *s, size_t n)
{
    uint64_t h = 1469598103934665603ull;
    for (size_t i = 0; i < n; i++) h = (h ^ (unsigned char)s[i]) * 1099511628211ull;
    return h;
}

static int vocab_get(const Tok *T, const char *s, size_t n)
{
    size_t m = T->vcap - 1, i = (size_t)fnv(s, n) & m;
    while (T->vk[i]) {
        if ((size_t)T->vlen[i] == n && !memcmp(T->vk[i], s, n)) return T->vid[i];
        i = (i + 1) & m;
    }
    return -1;
}
static void vocab_put(Tok *T, const char *s, size_t n, int id)
{
    size_t m = T->vcap - 1, i = (size_t)fnv(s, n) & m;
    while (T->vk[i]) i = (i + 1) & m;
    T->vk[i] = s;
    T->vlen[i] = (int)n;
    T->vid[i] = id;
}

static size_t mhash(const Tok *T, uint64_t key) { return (size_t)((key * 0x9E3779B97F4A7C15ull) >> 20) & (T->mcap - 1); }
static int merge_get(const Tok *T, int a, int b, int *newid)
{
    uint64_t key = ((uint64_t)(uint32_t)a << 32) | (uint32_t)b;
    size_t i = mhash(T, key);
    while (T->mkey[i] != ~0ull) {
        if (T->mkey[i] == key) {
            *newid = T->mnew[i];
            return T->mrank[i];
        }
        i = (i + 1) & (T->mcap - 1);
    }
    return -1;
}

static size_t pow2_at_least(size_t n)
{
    size_t p = 1;
    while (p < n) p <<= 1;
    return p;
}

static void tok_load(Tok *T, const char *path)
{
    memset(T, 0, sizeof *T);
    size_t len;
    char *txt = read_file(path, &len);
    JV *root = jparse(&T->A, txt, len);
    free(txt); /* strings were copied into the arena */

    const JV *model = jget(root, "model"), *pre = jget(root, "pre_tokenizer"), *norm = jget(root, "normalizer");
    const JV *mt = jget(model, "type"), *pt = jget(pre, "type");
    if (!model || !mt || strcmp(mt->s, "BPE")) die("%s: only BPE tokenizers are supported", path);
    if (!pt || strcmp(pt->s, "ByteLevel")) die("%s: only the ByteLevel pre-tokenizer is supported", path);
    const JV *ur = jget(pre, "use_regex"), *ps = jget(pre, "add_prefix_space");
    if ((ur && ur->t == J_FALSE) || (ps && ps->t == J_TRUE)) die("%s: unsupported ByteLevel options", path);
    const JV *nt = norm ? jget(norm, "type") : NULL;
    if (nt && strcmp(nt->s, "NFC")) fprintf(stderr, "laya: warning: normalizer %s is not implemented\n", nt->s);

    const JV *vocab = jget(model, "vocab"), *merges = jget(model, "merges");
    if (!vocab || vocab->t != J_OBJ || !merges || merges->t != J_ARR) die("%s: bad model section", path);
    T->vcap = pow2_at_least(vocab->n * 2 + 16);
    T->vk = xmalloc(T->vcap * sizeof(char *));
    memset(T->vk, 0, T->vcap * sizeof(char *));
    T->vlen = xmalloc(T->vcap * sizeof(int));
    T->vid = xmalloc(T->vcap * sizeof(int));
    for (size_t i = 0; i < vocab->n; i++) vocab_put(T, vocab->k[i], strlen(vocab->k[i]), (int)vocab->v[i]->num);

    /* GPT-2 byte <-> unicode map */
    uint32_t b2u[256];
    int extra = 0;
    for (int b = 0; b < 256; b++) {
        if ((b >= 33 && b <= 126) || (b >= 161 && b <= 172) || (b >= 174 && b <= 255)) b2u[b] = (uint32_t)b;
        else b2u[b] = (uint32_t)(256 + extra++);
        T->bchlen[b] = utf8_enc(b2u[b], T->bch[b]);
        T->byte2id[b] = vocab_get(T, T->bch[b], (size_t)T->bchlen[b]);
    }

    T->mcap = pow2_at_least(merges->n * 2 + 16);
    T->mkey = xmalloc(T->mcap * sizeof(uint64_t));
    for (size_t i = 0; i < T->mcap; i++) T->mkey[i] = ~0ull;
    T->mrank = xmalloc(T->mcap * sizeof(int));
    T->mnew = xmalloc(T->mcap * sizeof(int));
    for (size_t r = 0; r < merges->n; r++) {
        const JV *m = merges->v[r];
        const char *a, *b;
        size_t al, bl;
        if (m->t == J_STR) { /* "left right" */
            const char *sp = memchr(m->s, ' ', m->len);
            if (!sp) continue;
            a = m->s;
            al = (size_t)(sp - m->s);
            b = sp + 1;
            bl = m->len - al - 1;
        } else if (m->t == J_ARR && m->n == 2) { /* ["left", "right"] */
            a = m->v[0]->s;
            al = m->v[0]->len;
            b = m->v[1]->s;
            bl = m->v[1]->len;
        } else {
            continue;
        }
        int ia = vocab_get(T, a, al), ib = vocab_get(T, b, bl);
        char *cat = xmalloc(al + bl + 1);
        memcpy(cat, a, al);
        memcpy(cat + al, b, bl);
        int ic = vocab_get(T, cat, al + bl);
        free(cat);
        if (ia < 0 || ib < 0 || ic < 0) continue;
        uint64_t key = ((uint64_t)(uint32_t)ia << 32) | (uint32_t)ib;
        size_t i = mhash(T, key);
        int dup = 0;
        while (T->mkey[i] != ~0ull) {
            if (T->mkey[i] == key) {
                dup = 1;
                break;
            }
            i = (i + 1) & (T->mcap - 1);
        }
        if (dup) continue; /* keep the lowest rank */
        T->mkey[i] = key;
        T->mrank[i] = (int)r;
        T->mnew[i] = ic;
    }

    /* added tokens */
    const JV *at = jget(root, "added_tokens");
    T->cls = T->sep = T->mask = T->pad = -1;
    if (at && at->t == J_ARR) {
        for (int c = 0; c < 2; c++) T->add[c] = arena_alloc(&T->A, sizeof(AddTok) * (at->n + 1));
        for (size_t i = 0; i < at->n; i++) {
            const JV *e = at->v[i], *content = jget(e, "content"), *id = jget(e, "id"), *nm = jget(e, "normalized");
            const JV *ls = jget(e, "lstrip"), *rs = jget(e, "rstrip");
            if (!content || !id) continue;
            if (!strcmp(content->s, "[CLS]")) T->cls = (int)id->num;
            if (!strcmp(content->s, "[SEP]")) T->sep = (int)id->num;
            if (!strcmp(content->s, "[PAD]")) T->pad = (int)id->num;
            if (!strcmp(content->s, "[MASK]")) {
                T->mask = (int)id->num;
                T->mask_str = content->s;
            }
            if (content->len == 0) continue;
            int cls = (nm && nm->t == J_TRUE) ? 1 : 0;
            AddTok *a = &T->add[cls][T->nadd[cls]++];
            a->s = content->s;
            a->len = (int)content->len;
            a->id = (int)id->num;
            a->lstrip = ls && ls->t == J_TRUE;
            a->rstrip = rs && rs->t == J_TRUE;
            T->addfirst[cls][(unsigned char)content->s[0]] = 1;
        }
    }
    if (T->cls < 0 || T->sep < 0 || T->mask < 0) die("%s: [CLS]/[SEP]/[MASK] not found in added_tokens", path);
    if (!T->mask_str) T->mask_str = "[MASK]";
}

/* ---- Unicode classes for the GPT-2 pre-tokenizer regex: 0=other 1=letter(\p{L}) 2=number(\p{N}) 3=space(\s) ---- */

typedef struct {
    uint32_t lo, hi;
} R;
static const R R_NUM[] = {{0x660, 0x669}, {0x6F0, 0x6F9}, {0x7C0, 0x7C9}, {0xE50, 0xE59}, {0xED0, 0xED9}, {0xF20, 0xF33},
                          {0x1040, 0x1049}, {0x17E0, 0x17E9}, {0x1810, 0x1819}, {0x2070, 0x2070}, {0x2074, 0x2079},
                          {0x2080, 0x2089}, {0x2150, 0x2182}, {0x2185, 0x2189}, {0x2460, 0x249B}, {0x24EA, 0x24FF},
                          {0x2776, 0x2793}, {0x3007, 0x3007}, {0x3021, 0x3029}, {0x3038, 0x303A}, {0xFF10, 0xFF19},
                          {0x1D7CE, 0x1D7FF}};
static const R R_LET[] = {{0x2071, 0x2071}, {0x207F, 0x207F}, {0x2090, 0x209C}, {0x2102, 0x2102}, {0x2107, 0x2107},
                          {0x210A, 0x2113}, {0x2115, 0x2115}, {0x2119, 0x211D}, {0x2124, 0x2124}, {0x2126, 0x2126},
                          {0x2128, 0x2128}, {0x212A, 0x212D}, {0x212F, 0x2139}, {0x213C, 0x213F}, {0x2145, 0x2149},
                          {0x214E, 0x214E}, {0x2183, 0x2184}, {0x3005, 0x3006}, {0x303B, 0x303C}};
static const R R_OTH[] = {
    {0x2C2, 0x2C5},   {0x2D2, 0x2DF},   {0x2E5, 0x2EB},   {0x2ED, 0x2ED},   {0x2EF, 0x36F},   {0x375, 0x375},
    {0x37E, 0x37E},   {0x384, 0x385},   {0x387, 0x387},   {0x3F6, 0x3F6},   {0x482, 0x489},   {0x55A, 0x55F},
    {0x589, 0x58F},   {0x591, 0x5C7},   {0x5F3, 0x5F4},   {0x600, 0x61F},   {0x64B, 0x65F},   {0x66A, 0x66D},
    {0x670, 0x670},   {0x6D4, 0x6D4},   {0x6D6, 0x6E4},   {0x6E7, 0x6ED},   {0x6FD, 0x6FE},   {0x700, 0x70F},
    {0x711, 0x711},   {0x730, 0x74A},   {0xE31, 0xE31},   {0xE34, 0xE3F},   {0xE47, 0xE4F},   {0xE5A, 0xE5B},
    {0x1AB0, 0x1AFF}, {0x1DC0, 0x1DFF}, {0x2000, 0x2BFF}, {0x2E00, 0x2FFF}, {0x3000, 0x3004}, {0x3008, 0x3020},
    {0x302A, 0x3037}, {0x303D, 0x303F}, {0x3099, 0x309C}, {0x30A0, 0x30A0}, {0x30FB, 0x30FB}, {0x31C0, 0x31EF},
    {0x3200, 0x33FF}, {0x4DC0, 0x4DFF}, {0xA490, 0xA4CF}, {0xA4FE, 0xA4FF}, {0xA60D, 0xA60F}, {0xA670, 0xA67E},
    {0xA6F0, 0xA6F7}, {0xA700, 0xA716}, {0xA720, 0xA721}, {0xA789, 0xA78A}, {0xD800, 0xF8FF}, {0xFB1E, 0xFB1E},
    {0xFD3E, 0xFD3F}, {0xFDFC, 0xFDFF}, {0xFE00, 0xFE6F}, {0xFEFF, 0xFEFF}, {0xFF00, 0xFF0F}, {0xFF1A, 0xFF20},
    {0xFF3B, 0xFF40}, {0xFF5B, 0xFF65}, {0xFFE0, 0xFFFF}, {0x1D000, 0x1D3FF}, {0x1F000, 0x1FAFF}, {0x1FB00, 0x1FBEF},
    {0xE0000, 0xE0FFF}, {0xF0000, 0x10FFFF}};

static int in_ranges(const R *r, size_t n, uint32_t c)
{
    for (size_t i = 0; i < n; i++)
        if (c >= r[i].lo && c <= r[i].hi) return 1;
    return 0;
}

static int ucls(uint32_t c)
{
    if (c < 0x80) {
        if ((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z')) return 1;
        if (c >= '0' && c <= '9') return 2;
        if (c == ' ' || (c >= 9 && c <= 13)) return 3;
        return 0;
    }
    if (c == 0x85 || c == 0xA0 || c == 0x1680 || (c >= 0x2000 && c <= 0x200A) || c == 0x2028 || c == 0x2029 ||
        c == 0x202F || c == 0x205F || c == 0x3000)
        return 3;
    if (c < 0x100) {
        if (c == 0xAA || c == 0xB5 || c == 0xBA || (c >= 0xC0 && c <= 0xFF && c != 0xD7 && c != 0xF7)) return 1;
        if (c == 0xB2 || c == 0xB3 || c == 0xB9 || (c >= 0xBC && c <= 0xBE)) return 2;
        return 0;
    }
    if (in_ranges(R_NUM, sizeof R_NUM / sizeof *R_NUM, c)) return 2;
    if (in_ranges(R_LET, sizeof R_LET / sizeof *R_LET, c)) return 1;
    if (c >= 0x900 && c <= 0xDFF) { /* Indic blocks share one layout: signs/vowel signs/danda are marks or punctuation */
        uint32_t o = c & 0x7F;
        if (o >= 0x66 && o <= 0x6F) return 2;
        if (o <= 3 || (o >= 0x3A && o <= 0x4F && o != 0x3D) || (o >= 0x51 && o <= 0x57) || o == 0x62 || o == 0x63 ||
            o == 0x64 || o == 0x65 || o == 0x70)
            return 0;
        return 1;
    }
    if (in_ranges(R_OTH, sizeof R_OTH / sizeof *R_OTH, c)) return 0;
    return 1;
}

/* Length end of the next pre-token starting at i (GPT-2 regex:
 * 's|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+ ), operating on [i, b). */
static size_t next_pretoken(const unsigned char *s, size_t i, size_t b)
{
    if (s[i] == '\'' && i + 1 < b) {
        unsigned char c = s[i + 1];
        if (c == 's' || c == 't' || c == 'm' || c == 'd') return i + 2;
        if (i + 2 < b) {
            unsigned char d = s[i + 2];
            if ((c == 'r' && d == 'e') || (c == 'v' && d == 'e') || (c == 'l' && d == 'l')) return i + 3;
        }
    }
    size_t st = (s[i] == ' ' && i + 1 < b) ? i + 1 : i;
    int n;
    uint32_t cp = utf8_dec(s, st, b, &n);
    int cl = ucls(cp);
    if (cl != 3) { /* letters / numbers / other: consume the run of the same class */
        size_t j = st + (size_t)n;
        while (j < b) {
            uint32_t c2 = utf8_dec(s, j, b, &n);
            if (ucls(c2) != cl) break;
            j += (size_t)n;
        }
        return j;
    }
    /* whitespace run starting at i */
    size_t j = i, last = i;
    int cnt = 0;
    while (j < b) {
        uint32_t c2 = utf8_dec(s, j, b, &n);
        if (ucls(c2) != 3) break;
        last = j;
        j += (size_t)n;
        cnt++;
    }
    if (j >= b) return j;   /* \s+ at end of text */
    if (cnt >= 2) return last; /* \s+(?!\S): leave the last space for the next token */
    return j;               /* single space before a non-space: plain \s+ */
}

typedef struct {
    int id, prev, next;
} Sym;
typedef struct {
    uint32_t rank;
    int pos, lid, rid, nid;
} HItem;

static int hless(const HItem *a, const HItem *b) { return a->rank < b->rank || (a->rank == b->rank && a->pos < b->pos); }
static void hpush(HItem **h, size_t *n, size_t *cap, HItem it)
{
    if (*n == *cap) {
        *cap = *cap ? *cap * 2 : 64;
        *h = xrealloc(*h, *cap * sizeof(HItem));
    }
    size_t i = (*n)++;
    (*h)[i] = it;
    while (i > 0) {
        size_t p = (i - 1) / 2;
        if (!hless(&(*h)[i], &(*h)[p])) break;
        HItem t = (*h)[i];
        (*h)[i] = (*h)[p];
        (*h)[p] = t;
        i = p;
    }
}
static HItem hpop(HItem *h, size_t *n)
{
    HItem top = h[0];
    h[0] = h[--(*n)];
    size_t i = 0;
    for (;;) {
        size_t l = 2 * i + 1, r = l + 1, m = i;
        if (l < *n && hless(&h[l], &h[m])) m = l;
        if (r < *n && hless(&h[r], &h[m])) m = r;
        if (m == i) break;
        HItem t = h[i];
        h[i] = h[m];
        h[m] = t;
        i = m;
    }
    return top;
}

static void bpe_word(const Tok *T, const unsigned char *w, size_t n, IVec *out)
{
    Sym *sy = xmalloc(n * sizeof(Sym));
    HItem *heap = NULL;
    size_t hn = 0, hcap = 0;
    for (size_t i = 0; i < n; i++) {
        sy[i].id = T->byte2id[w[i]];
        sy[i].prev = (int)i - 1;
        sy[i].next = i + 1 < n ? (int)i + 1 : -1;
    }
    for (size_t i = 0; i + 1 < n; i++) {
        int nid, r = merge_get(T, sy[i].id, sy[i + 1].id, &nid);
        if (r >= 0) hpush(&heap, &hn, &hcap, (HItem){(uint32_t)r, (int)i, sy[i].id, sy[i + 1].id, nid});
    }
    while (hn) {
        HItem it = hpop(heap, &hn);
        int l = it.pos, r = sy[l].next;
        if (sy[l].id != it.lid || r < 0 || sy[r].id != it.rid) continue; /* stale entry */
        sy[l].id = it.nid;
        sy[l].next = sy[r].next;
        if (sy[r].next >= 0) sy[sy[r].next].prev = l;
        sy[r].id = -1;
        int nid, rk;
        if (sy[l].prev >= 0 && (rk = merge_get(T, sy[sy[l].prev].id, sy[l].id, &nid)) >= 0)
            hpush(&heap, &hn, &hcap, (HItem){(uint32_t)rk, sy[l].prev, sy[sy[l].prev].id, sy[l].id, nid});
        if (sy[l].next >= 0 && (rk = merge_get(T, sy[l].id, sy[sy[l].next].id, &nid)) >= 0)
            hpush(&heap, &hn, &hcap, (HItem){(uint32_t)rk, l, sy[l].id, sy[sy[l].next].id, nid});
    }
    for (int i = 0; i >= 0; i = sy[i].next)
        if (sy[i].id >= 0) iv_push(out, sy[i].id);
    free(heap);
    free(sy);
}

typedef struct {
    size_t a, b;
    int id;
} Seg;
typedef struct {
    Seg *s;
    size_t n, cap;
} SegV;
static void sg_push(SegV *v, size_t a, size_t b, int id)
{
    if (v->n == v->cap) {
        v->cap = v->cap ? v->cap * 2 : 16;
        v->s = xrealloc(v->s, v->cap * sizeof(Seg));
    }
    v->s[v->n++] = (Seg){a, b, id};
}

/* Split text on added tokens (leftmost-longest), like tokenizers' AddedVocabulary. */
static void split_added(const Tok *T, int cls, const char *text, const SegV *in, SegV *out)
{
    for (size_t k = 0; k < in->n; k++) {
        Seg sg = in->s[k];
        if (sg.id >= 0 || T->nadd[cls] == 0) {
            sg_push(out, sg.a, sg.b, sg.id);
            continue;
        }
        size_t pos = sg.a, last = sg.a;
        while (pos < sg.b) {
            unsigned char c = (unsigned char)text[pos];
            if (T->addfirst[cls][c]) {
                int best = -1;
                for (int j = 0; j < T->nadd[cls]; j++) {
                    const AddTok *t = &T->add[cls][j];
                    if ((unsigned char)t->s[0] == c && pos + (size_t)t->len <= sg.b && !memcmp(text + pos, t->s, (size_t)t->len) &&
                        (best < 0 || t->len > T->add[cls][best].len))
                        best = j;
                }
                if (best >= 0) {
                    const AddTok *t = &T->add[cls][best];
                    size_t start = pos;
                    while (t->lstrip && start > last && strchr(" \t\n\r\v\f", text[start - 1])) start--;
                    if (last < start) sg_push(out, last, start, -1);
                    sg_push(out, start, pos + (size_t)t->len, t->id);
                    pos += (size_t)t->len;
                    while (t->rstrip && pos < sg.b && text[pos] && strchr(" \t\n\r\v\f", text[pos])) pos++;
                    last = pos;
                    continue;
                }
            }
            pos++;
        }
        if (last < sg.b) sg_push(out, last, sg.b, -1);
    }
}

/* Encode without special tokens (add_special_tokens=False). Stops early once >= limit ids exist
 * (callers only ever keep a prefix), limit < 0 means unlimited. */
static void tok_encode(const Tok *T, const char *text, size_t len, IVec *out, int limit)
{
    SegV a = {0}, b = {0};
    sg_push(&a, 0, len, -1);
    split_added(T, 0, text, &a, &b);
    a.n = 0;
    split_added(T, 1, text, &b, &a);
    for (size_t k = 0; k < a.n; k++) {
        if (limit >= 0 && out->n >= limit) break;
        Seg sg = a.s[k];
        if (sg.id >= 0) {
            iv_push(out, sg.id);
            continue;
        }
        const unsigned char *s = (const unsigned char *)text;
        /* per pre-token: map bytes -> byte-level symbols and run BPE */
        size_t i = sg.a;
        while (i < sg.b) {
            size_t e = next_pretoken(s, i, sg.b);
            bpe_word(T, s + i, e - i, out);
            i = e;
            if (limit >= 0 && out->n >= limit) break;
        }
    }
    free(a.s);
    free(b.s);
}

/* ================================================================== model */

typedef struct {
    float *attn_norm; /* NULL on layer 0 (Identity) */
    float *Wqkv, *Wo, *mlp_norm, *Wi, *Wo2;
    int local;
} EncLayer;

typedef struct {
    float *n1w, *n1b, *n2w, *n2b, *in_w, *in_b, *out_w, *out_b, *l1w, *l1b, *l2w, *l2b;
} HeadLayer;

typedef struct {
    char key[24];
    float t;
} TempEntry;

typedef struct Laya {
    Tok tok;
    int have_model;
    /* encoder */
    int d, nl, nh, dff, vocab, window, glob_every;
    float eps, theta_g, theta_l;
    float *emb, *emb_norm, *final_norm;
    EncLayer *L;
    /* head */
    int nhl, hh, hff;
    HeadLayer *H;
    float *type_emb;
    float *sc_lnw, *sc_lnb, *sc_w1, *sc_b1, *sc_w2, *sc_b2;
    int act_hidden, n_act;
    float *act_w1, *act_b1, *act_w2, *act_b2;
    /* config */
    int max_len, head_max_len;
    float temperature[3];
    TempEntry temps[16];
    int ntemps;
} Laya;

static float *ld(STFile *st, const char *name, size_t numel, int enc)
{
    static const char *pref[] = {"encoder.", "model.", ""};
    char buf[256];
    const STensor *T = NULL;
    if (enc) {
        for (int i = 0; i < 3 && !T; i++) {
            snprintf(buf, sizeof buf, "%s%s", pref[i], name);
            T = st_find(st, buf);
        }
    } else {
        T = st_find(st, name);
    }
    if (!T) die("tensor '%s' not found in checkpoint (run --list-tensors to see what it contains)", name);
    return st_load(st, T, numel);
}

/* Register-tile geometry shared by the weight packer and linear(). */
#if (defined(__GNUC__) || defined(__clang__)) && !defined(LAYA_NO_SIMD)
#define LAYA_VEC 1
#if defined(__AVX512F__)
#define VW 16
#else
#define VW 8
#endif
typedef float vf __attribute__((vector_size(VW * 4), aligned(4), may_alias));
#else
#define VW 8
#endif
#define MR 6
#define NR (2 * VW)
#define KB 256

/* Load an nn.Linear weight [N,K] and repack it into NR-column panels: panel p holds columns p*NR..p*NR+NR-1 as
 * [K][NR] contiguous floats (zero padded), so linear() streams weights sequentially. */
static float *ldT(STFile *st, const char *name, size_t N, size_t K, int enc)
{
    float *w = ld(st, name, N * K, enc);
    size_t npan = (N + NR - 1) / NR;
    float *t = xcalloc(npan * K * NR, sizeof(float));
    for (size_t n = 0; n < N; n++) {
        float *dst = t + (n / NR) * K * NR + n % NR;
        const float *src = w + n * K;
        for (size_t k = 0; k < K; k++) dst[k * NR] = src[k];
    }
    free(w);
    return t;
}

static double cfg_num(const JV *o, const char *k, double dflt)
{
    const JV *v = jget(o, k);
    return v && v->t == J_NUM ? v->num : dflt;
}

static void load_config(Laya *M, const char *dir, const JV **encfg_out, Arena *A)
{
    char path[1024];
    size_t len;
    snprintf(path, sizeof path, "%s/rl_agent_config.json", dir);
    char *txt = read_file(path, &len);
    JV *cfg = jparse(A, txt, len);
    free(txt);
    M->max_len = (int)cfg_num(cfg, "max_len", 512);
    M->head_max_len = (int)cfg_num(cfg, "head_max_len", 192);
    M->nhl = (int)cfg_num(cfg, "head_layers", 2);
    const JV *tm = jget(cfg, "temperature");
    for (int i = 0; i < 3; i++) M->temperature[i] = (tm && tm->t == J_ARR && (size_t)i < tm->n) ? (float)tm->v[i]->num : 1.0f;
    const JV *tb = jget(cfg, "temperature_by_options");
    M->ntemps = 0;
    if (tb && tb->t == J_OBJ)
        for (size_t i = 0; i < tb->n && M->ntemps < 16; i++) {
            snprintf(M->temps[M->ntemps].key, sizeof M->temps[0].key, "%s", tb->k[i]);
            M->temps[M->ntemps++].t = (float)tb->v[i]->num;
        }

    /* encoder architecture: encoder/config.json (falls back to ModernBERT-large defaults) */
    snprintf(path, sizeof path, "%s/config.json", dir);
    FILE *f = fopen(path, "rb");
    const JV *ec = NULL;
    if (f) {
        fclose(f);
        txt = read_file(path, &len);
        ec = jparse(A, txt, len);
        free(txt);
    } else {
        fprintf(stderr, "laya: %s missing, assuming ModernBERT-large hyper-parameters\n", path);
    }
    M->d = (int)cfg_num(ec, "hidden_size", 1024);
    M->nl = (int)cfg_num(ec, "num_hidden_layers", 28);
    M->nh = (int)cfg_num(ec, "num_attention_heads", 16);
    M->dff = (int)cfg_num(ec, "intermediate_size", 2624);
    M->vocab = (int)cfg_num(ec, "vocab_size", 50368);
    M->window = (int)cfg_num(ec, "local_attention", 128) / 2;
    M->glob_every = (int)cfg_num(ec, "global_attn_every_n_layers", 3);
    M->eps = (float)cfg_num(ec, "norm_eps", cfg_num(ec, "layer_norm_eps", 1e-5));
    M->theta_g = (float)cfg_num(ec, "global_rope_theta", 160000.0);
    M->theta_l = (float)cfg_num(ec, "local_rope_theta", 10000.0);
    const JV *rp = jget(ec, "rope_parameters");
    if (rp) {
        const JV *fa = jget(jget(rp, "full_attention"), "rope_theta"), *sw = jget(jget(rp, "sliding_attention"), "rope_theta");
        if (fa) M->theta_g = (float)fa->num;
        if (sw) M->theta_l = (float)sw->num;
    }
    const JV *ha = jget(ec, "hidden_activation");
    if (ha && ha->t == J_STR && strcmp(ha->s, "gelu")) die("hidden_activation '%s' is not supported (only gelu)", ha->s);
    if (M->d % M->nh || (M->d / M->nh) % 2) die("bad head geometry in encoder config");
    if (M->d / M->nh > 256) die("head_dim > 256 is not supported");
    *encfg_out = ec;
}

static Laya *laya_load_impl(const char *dir, int tokenizer_only, int list_only)
{
    Laya *M = xmalloc(sizeof *M);
    memset(M, 0, sizeof *M);
    char path[1024];
    snprintf(path, sizeof path, "%s/tokenizer.json", dir);
    if (!list_only) tok_load(&M->tok, path);
    if (tokenizer_only) return M;

    Arena A = {0};
    const JV *ec;
    load_config(M, dir, &ec, &A);
    const JV *lt = jget(ec, "layer_types");
    STFile st;
    snprintf(path, sizeof path, "%s/model.safetensors", dir);
    st_open(&st, path);
    if (list_only) {
        for (int i = 0; i < st.n; i++) {
            printf("%s  %s [", st.t[i].name, st.t[i].dtype == 0 ? "F32" : st.t[i].dtype == 1 ? "F16" : "BF16");
            for (int d = 0; d < st.t[i].ndim; d++) printf(d ? ",%lld" : "%lld", st.t[i].shape[d]);
            puts("]");
        }
        return M;
    }

    size_t d = (size_t)M->d, dff = (size_t)M->dff;
    M->emb = ld(&st, "embeddings.tok_embeddings.weight", (size_t)M->vocab * d, 1);
    M->emb_norm = ld(&st, "embeddings.norm.weight", d, 1);
    M->final_norm = ld(&st, "final_norm.weight", d, 1);
    M->L = xmalloc(sizeof(EncLayer) * (size_t)M->nl);
    for (int i = 0; i < M->nl; i++) {
        char nm[96];
        EncLayer *L = &M->L[i];
        if (lt && lt->t == J_ARR && (size_t)i < lt->n) L->local = strcmp(lt->v[i]->s, "sliding_attention") == 0;
        else L->local = (i % M->glob_every) != 0;
        if (i == 0) L->attn_norm = NULL;
        else {
            snprintf(nm, sizeof nm, "layers.%d.attn_norm.weight", i);
            L->attn_norm = ld(&st, nm, d, 1);
        }
        snprintf(nm, sizeof nm, "layers.%d.attn.Wqkv.weight", i);
        L->Wqkv = ldT(&st, nm, 3 * d, d, 1);
        snprintf(nm, sizeof nm, "layers.%d.attn.Wo.weight", i);
        L->Wo = ldT(&st, nm, d, d, 1);
        snprintf(nm, sizeof nm, "layers.%d.mlp_norm.weight", i);
        L->mlp_norm = ld(&st, nm, d, 1);
        snprintf(nm, sizeof nm, "layers.%d.mlp.Wi.weight", i);
        L->Wi = ldT(&st, nm, 2 * dff, d, 1);
        snprintf(nm, sizeof nm, "layers.%d.mlp.Wo.weight", i);
        L->Wo2 = ldT(&st, nm, d, dff, 1);
    }
    /* decision head: nn.TransformerEncoder(norm_first=True, relu, nhead=d//64, ff=4d) */
    M->hh = M->d / 64 > 1 ? M->d / 64 : 1;
    const STensor *l1 = st_find(&st, "head.layers.0.linear1.weight");
    if (M->nhl > 0 && !l1) die("head.layers.0.linear1.weight not found (head_layers=%d in config)", M->nhl);
    M->hff = l1 ? (int)l1->shape[0] : 4 * M->d;
    size_t hff = (size_t)M->hff;
    M->H = xmalloc(sizeof(HeadLayer) * (size_t)(M->nhl > 0 ? M->nhl : 1));
    for (int i = 0; i < M->nhl; i++) {
        char nm[96];
        HeadLayer *H = &M->H[i];
#define HL(field, suffix, n)                                                                                             \
    snprintf(nm, sizeof nm, "head.layers.%d.%s", i, suffix);                                                             \
    H->field = ld(&st, nm, n, 0);
#define HT(field, suffix, N, K)                                                                                          \
    snprintf(nm, sizeof nm, "head.layers.%d.%s", i, suffix);                                                             \
    H->field = ldT(&st, nm, N, K, 0);
        HL(n1w, "norm1.weight", d) HL(n1b, "norm1.bias", d) HL(n2w, "norm2.weight", d) HL(n2b, "norm2.bias", d)
        HT(in_w, "self_attn.in_proj_weight", 3 * d, d) HL(in_b, "self_attn.in_proj_bias", 3 * d)
        HT(out_w, "self_attn.out_proj.weight", d, d) HL(out_b, "self_attn.out_proj.bias", d)
        HT(l1w, "linear1.weight", hff, d) HL(l1b, "linear1.bias", hff)
        HT(l2w, "linear2.weight", d, hff) HL(l2b, "linear2.bias", d)
#undef HL
#undef HT
    }
    M->type_emb = ld(&st, "type_emb.weight", 3 * d, 0);
    M->sc_lnw = ld(&st, "scorer.0.weight", d, 0);
    M->sc_lnb = ld(&st, "scorer.0.bias", d, 0);
    M->sc_w1 = ldT(&st, "scorer.1.weight", d, d, 0);
    M->sc_b1 = ld(&st, "scorer.1.bias", d, 0);
    M->sc_w2 = ld(&st, "scorer.3.weight", d, 0);
    M->sc_b2 = ld(&st, "scorer.3.bias", 1, 0);
    const STensor *a0 = st_find(&st, "act_head.0.weight"), *a2 = st_find(&st, "act_head.2.weight");
    if (!a0 || !a2) die("act_head tensors not found");
    M->act_hidden = (int)a0->shape[0];
    M->n_act = (int)a2->shape[0];
    if (M->n_act > 16) die("act_head has %d outputs (max 16)", M->n_act);
    M->act_w1 = ldT(&st, "act_head.0.weight", (size_t)M->act_hidden, d + 4, 0);
    M->act_b1 = ld(&st, "act_head.0.bias", (size_t)M->act_hidden, 0);
    M->act_w2 = ldT(&st, "act_head.2.weight", (size_t)M->n_act, (size_t)M->act_hidden, 0);
    M->act_b2 = ld(&st, "act_head.2.bias", (size_t)M->n_act, 0);
    fclose(st.f);
    for (int i = 0; i < st.n; i++) free(st.t[i].name);
    free(st.t);
    arena_free(&A);
    M->have_model = 1;
    return M;
}

Laya *laya_load(const char *model_dir) { return laya_load_impl(model_dir, 0, 0); }

void laya_free(Laya *M)
{
    if (!M) return;
    if (M->have_model) {
        free(M->emb);
        free(M->emb_norm);
        free(M->final_norm);
        for (int i = 0; i < M->nl; i++) {
            EncLayer *L = &M->L[i];
            free(L->attn_norm);
            free(L->Wqkv);
            free(L->Wo);
            free(L->mlp_norm);
            free(L->Wi);
            free(L->Wo2);
        }
        free(M->L);
        for (int i = 0; i < M->nhl; i++) {
            HeadLayer *H = &M->H[i];
            float *p[] = {H->n1w, H->n1b, H->n2w, H->n2b, H->in_w, H->in_b, H->out_w, H->out_b, H->l1w, H->l1b, H->l2w, H->l2b};
            for (size_t k = 0; k < sizeof p / sizeof *p; k++) free(p[k]);
        }
        free(M->H);
        float *q[] = {M->type_emb, M->sc_lnw, M->sc_lnb, M->sc_w1, M->sc_b1, M->sc_w2, M->sc_b2, M->act_w1, M->act_b1, M->act_w2, M->act_b2};
        for (size_t k = 0; k < sizeof q / sizeof *q; k++) free(q[k]);
    }
    free(M->tok.vk);
    free(M->tok.vlen);
    free(M->tok.vid);
    free(M->tok.mkey);
    free(M->tok.mrank);
    free(M->tok.mnew);
    arena_free(&M->tok.A);
    free(M);
}

/* ================================================================== math kernels */

static float dotf(const float *a, const float *b, int n)
{
    float s[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    int k = 0;
    for (; k + 8 <= n; k += 8)
        for (int u = 0; u < 8; u++) s[u] += a[k + u] * b[k + u];
    float t = ((s[0] + s[4]) + (s[1] + s[5])) + ((s[2] + s[6]) + (s[3] + s[7]));
    for (; k < n; k++) t += a[k] * b[k];
    return t;
}

/* C[M,N] = A[M,K] . W (+ bias[N]) with W packed by ldT(). MR x NR register tile, K walked in blocks of KB whose
 * partial sums are added in a fixed order, so results do not depend on the thread count. */
static void linear(float *restrict C, const float *restrict A, int M, int K, const float *restrict Wp, const float *restrict bias, int N)
{
    int npan = (N + NR - 1) / NR;
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (int nb = 0; nb < npan; nb++) {
        const float *wp = Wp + (size_t)nb * K * NR;
        int n0 = nb * NR, nr = N - n0 < NR ? N - n0 : NR;
        for (int m0 = 0; m0 < M; m0 += MR) {
            int mr = M - m0 < MR ? M - m0 : MR;
            const float *ar[MR];
            for (int i = 0; i < MR; i++) ar[i] = A + (size_t)(m0 + (i < mr ? i : 0)) * K; /* pad rows: result discarded */
            float tot[MR][NR];
            memset(tot, 0, sizeof tot);
            for (int k0 = 0; k0 < K; k0 += KB) {
                int k1 = K - k0 < KB ? K : k0 + KB;
#ifdef LAYA_VEC
                vf c[MR][2], z = {0};
                for (int i = 0; i < MR; i++) c[i][0] = c[i][1] = z;
                for (int k = k0; k < k1; k++) {
                    vf w0 = *(const vf *)(wp + (size_t)k * NR), w1 = *(const vf *)(wp + (size_t)k * NR + VW);
                    for (int i = 0; i < MR; i++) {
                        float x = ar[i][k];
                        c[i][0] += w0 * x;
                        c[i][1] += w1 * x;
                    }
                }
                for (int i = 0; i < MR; i++) {
                    *(vf *)&tot[i][0] += c[i][0];
                    *(vf *)&tot[i][VW] += c[i][1];
                }
#else
                float c[MR][NR] = {{0}};
                for (int k = k0; k < k1; k++)
                    for (int i = 0; i < MR; i++)
                        for (int j = 0; j < NR; j++) c[i][j] += ar[i][k] * wp[(size_t)k * NR + j];
                for (int i = 0; i < MR; i++)
                    for (int j = 0; j < NR; j++) tot[i][j] += c[i][j];
#endif
            }
            for (int i = 0; i < mr; i++)
                for (int j = 0; j < nr; j++) C[(size_t)(m0 + i) * N + n0 + j] = tot[i][j] + (bias ? bias[n0 + j] : 0.0f);
        }
    }
}

static void layernorm(float *out, const float *x, const float *w, const float *b, int T, int d, float eps)
{
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (int t = 0; t < T; t++) {
        const float *xr = x + (size_t)t * d;
        float *o = out + (size_t)t * d;
        double mu = 0, var = 0;
        for (int i = 0; i < d; i++) mu += xr[i];
        mu /= d;
        for (int i = 0; i < d; i++) {
            double c = xr[i] - mu;
            var += c * c;
        }
        var /= d;
        float inv = (float)(1.0 / sqrt(var + (double)eps)), m = (float)mu;
        for (int i = 0; i < d; i++) o[i] = (xr[i] - m) * inv * w[i] + (b ? b[i] : 0.0f);
    }
}

static inline float gelu(float x) { return 0.5f * x * (1.0f + erff(x * 0.70710678118654752f)); }

/* Multi-head attention over qkv[T, 3d] laid out [q | k | v], head-major inside each.
 * win >= 0: bidirectional sliding window |i-j| <= win; win < 0: full. cs/sn: RoPE tables [T, hd/2] or NULL. */
static void mha(float *ao, const float *qkv, int T, int d, int nh, int win, const float *cs, const float *sn)
{
    int hd = d / nh, half = hd / 2;
    float scale = 1.0f / sqrtf((float)hd);
    size_t per = (size_t)nh * T * hd;
    float *Q = xmalloc(3 * per * sizeof(float)), *K = Q + per, *V = K + per;
    for (int t = 0; t < T; t++)
        for (int h = 0; h < nh; h++) {
            const float *q = qkv + (size_t)t * 3 * d + (size_t)h * hd, *k = q + d, *v = q + 2 * d;
            float *qo = Q + ((size_t)h * T + t) * hd, *ko = K + ((size_t)h * T + t) * hd, *vo = V + ((size_t)h * T + t) * hd;
            if (cs) {
                const float *c = cs + (size_t)t * half, *s = sn + (size_t)t * half;
                for (int j = 0; j < half; j++) { /* rotate_half: pairs (j, j + hd/2) */
                    qo[j] = q[j] * c[j] - q[j + half] * s[j];
                    qo[j + half] = q[j + half] * c[j] + q[j] * s[j];
                    ko[j] = k[j] * c[j] - k[j + half] * s[j];
                    ko[j + half] = k[j + half] * c[j] + k[j] * s[j];
                }
            } else {
                memcpy(qo, q, sizeof(float) * hd);
                memcpy(ko, k, sizeof(float) * hd);
            }
            memcpy(vo, v, sizeof(float) * hd);
        }
#ifdef _OPENMP
#pragma omp parallel
#endif
    {
        float *sc = xmalloc(sizeof(float) * (size_t)T);
#ifdef _OPENMP
#pragma omp for collapse(2) schedule(static)
#endif
        for (int h = 0; h < nh; h++)
            for (int i = 0; i < T; i++) {
                int lo = win >= 0 && i - win > 0 ? i - win : 0;
                int hi = win >= 0 && i + win < T - 1 ? i + win : T - 1;
                const float *q = Q + ((size_t)h * T + i) * hd;
                float mx = -INFINITY;
                for (int j = lo; j <= hi; j++) {
                    float s = dotf(q, K + ((size_t)h * T + j) * hd, hd) * scale;
                    sc[j - lo] = s;
                    if (s > mx) mx = s;
                }
                float sum = 0;
                for (int j = lo; j <= hi; j++) {
                    sc[j - lo] = expf(sc[j - lo] - mx);
                    sum += sc[j - lo];
                }
                float inv = 1.0f / sum, acc[256];
                for (int c = 0; c < hd; c++) acc[c] = 0;
                for (int j = lo; j <= hi; j++) {
                    float p = sc[j - lo] * inv;
                    const float *v = V + ((size_t)h * T + j) * hd;
                    for (int c = 0; c < hd; c++) acc[c] += p * v[c];
                }
                memcpy(ao + (size_t)i * d + (size_t)h * hd, acc, sizeof(float) * hd);
            }
        free(sc);
    }
    free(Q);
}

static void rope_tables(float *cs, float *sn, int T, int hd, double theta)
{
    int half = hd / 2;
    for (int j = 0; j < half; j++) {
        double inv = 1.0 / pow(theta, (double)(2 * j) / hd);
        for (int t = 0; t < T; t++) {
            cs[(size_t)t * half + j] = (float)cos(t * inv);
            sn[(size_t)t * half + j] = (float)sin(t * inv);
        }
    }
}

/* DecisionModel.forward for one question: token ids + marker positions -> marker logits, act probability. */
static void forward(const Laya *M, const int *ids, int T, const int *markers, int K, int qtype, float *logits, float *act_prob)
{
    int d = M->d, dff = M->dff;
    size_t Td = (size_t)T * d;
    float *h = xmalloc(Td * 4 * sizeof(float)), *xn = h + Td, *ao = xn + Td, *tmp = ao + Td;
    float *qkv = xmalloc((size_t)T * 3 * d * sizeof(float));
    float *wi = xmalloc((size_t)T * 2 * dff * sizeof(float)), *g = xmalloc((size_t)T * dff * sizeof(float));
    int hd = d / M->nh;
    float *cg = xmalloc(sizeof(float) * (size_t)T * hd / 2 * 4), *sg = cg + (size_t)T * hd / 2, *cl = sg + (size_t)T * hd / 2, *sl = cl + (size_t)T * hd / 2;
    rope_tables(cg, sg, T, hd, M->theta_g);
    rope_tables(cl, sl, T, hd, M->theta_l);

    for (int t = 0; t < T; t++) {
        if (ids[t] < 0 || ids[t] >= M->vocab) die("token id %d outside embedding table", ids[t]);
        memcpy(h + (size_t)t * d, M->emb + (size_t)ids[t] * d, sizeof(float) * d);
    }
    layernorm(h, h, M->emb_norm, NULL, T, d, M->eps);

    for (int i = 0; i < M->nl; i++) {
        const EncLayer *L = &M->L[i];
        const float *in = h;
        if (L->attn_norm) {
            layernorm(xn, h, L->attn_norm, NULL, T, d, M->eps);
            in = xn;
        }
        linear(qkv, in, T, d, L->Wqkv, NULL, 3 * d);
        mha(ao, qkv, T, d, M->nh, L->local ? M->window : -1, L->local ? cl : cg, L->local ? sl : sg);
        linear(tmp, ao, T, d, L->Wo, NULL, d);
        for (size_t k = 0; k < Td; k++) h[k] += tmp[k];
        layernorm(xn, h, L->mlp_norm, NULL, T, d, M->eps);
        linear(wi, xn, T, d, L->Wi, NULL, 2 * dff);
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
        for (int t = 0; t < T; t++) {
            const float *r = wi + (size_t)t * 2 * dff;
            float *o = g + (size_t)t * dff;
            for (int j = 0; j < dff; j++) o[j] = gelu(r[j]) * r[dff + j]; /* act(input) * gate */
        }
        linear(tmp, g, T, dff, L->Wo2, NULL, d);
        for (size_t k = 0; k < Td; k++) h[k] += tmp[k];
    }
    layernorm(h, h, M->final_norm, NULL, T, d, M->eps);

    const float *te = M->type_emb + (size_t)qtype * d;
    for (int t = 0; t < T; t++)
        for (int j = 0; j < d; j++) h[(size_t)t * d + j] += te[j];

    float *hf = xmalloc((size_t)T * M->hff * sizeof(float));
    float *qkv2 = qkv;
    for (int i = 0; i < M->nhl; i++) {
        const HeadLayer *H = &M->H[i];
        layernorm(xn, h, H->n1w, H->n1b, T, d, 1e-5f);
        linear(qkv2, xn, T, d, H->in_w, H->in_b, 3 * d);
        mha(ao, qkv2, T, d, M->hh, -1, NULL, NULL);
        linear(tmp, ao, T, d, H->out_w, H->out_b, d);
        for (size_t k = 0; k < Td; k++) h[k] += tmp[k];
        layernorm(xn, h, H->n2w, H->n2b, T, d, 1e-5f);
        linear(hf, xn, T, d, H->l1w, H->l1b, M->hff);
        for (size_t k = 0; k < (size_t)T * M->hff; k++) hf[k] = hf[k] > 0 ? hf[k] : 0;
        linear(tmp, hf, T, M->hff, H->l2w, H->l2b, d);
        for (size_t k = 0; k < Td; k++) h[k] += tmp[k];
    }
    free(hf);

    /* option-marker scorer: LayerNorm -> Linear -> GELU -> Linear(d,1) at each marker */
    float *mrows = xmalloc(sizeof(float) * (size_t)K * d), *mn = xmalloc(sizeof(float) * (size_t)K * d), *m1 = xmalloc(sizeof(float) * (size_t)K * d);
    for (int k = 0; k < K; k++) memcpy(mrows + (size_t)k * d, h + (size_t)markers[k] * d, sizeof(float) * d);
    layernorm(mn, mrows, M->sc_lnw, M->sc_lnb, K, d, 1e-5f);
    linear(m1, mn, K, d, M->sc_w1, M->sc_b1, d);
    for (int k = 0; k < K; k++) {
        for (int j = 0; j < d; j++) m1[(size_t)k * d + j] = gelu(m1[(size_t)k * d + j]);
        logits[k] = dotf(m1 + (size_t)k * d, M->sc_w2, d) + M->sc_b2[0];
    }

    /* act/escalate head on the CLS state + [top1, top1-top2, normalized entropy, k/255] of the raw distribution */
    double mx = logits[0];
    for (int k = 1; k < K; k++)
        if (logits[k] > mx) mx = logits[k];
    double sum = 0, p[1024];
    if (K > 1024) die("too many options (%d)", K);
    for (int k = 0; k < K; k++) {
        p[k] = exp(logits[k] - mx);
        sum += p[k];
    }
    double t1 = 0, t2 = 0, ent = 0;
    for (int k = 0; k < K; k++) {
        p[k] /= sum;
        if (p[k] > t1) {
            t2 = t1;
            t1 = p[k];
        } else if (p[k] > t2) t2 = p[k];
        ent -= p[k] * log(p[k] > 1e-9 ? p[k] : 1e-9);
    }
    ent /= log((double)(K < 2 ? 2 : K));
    float *ain = xmalloc(sizeof(float) * (size_t)(d + 4)), *ah = xmalloc(sizeof(float) * (size_t)M->act_hidden), al[16];
    memcpy(ain, h, sizeof(float) * d); /* CLS position, after the head layers */
    ain[d] = (float)t1;
    ain[d + 1] = (float)(t1 - t2);
    ain[d + 2] = (float)ent;
    ain[d + 3] = (float)K / 255.0f;
    linear(ah, ain, 1, d + 4, M->act_w1, M->act_b1, M->act_hidden);
    for (int j = 0; j < M->act_hidden; j++) ah[j] = gelu(ah[j]);
    linear(al, ah, 1, M->act_hidden, M->act_w2, M->act_b2, M->n_act);
    float amx = al[0], asum = 0;
    for (int j = 1; j < M->n_act && j < 16; j++)
        if (al[j] > amx) amx = al[j];
    for (int j = 0; j < M->n_act && j < 16; j++) asum += expf(al[j] - amx);
    *act_prob = expf(al[0] - amx) / asum;

    free(ain);
    free(ah);
    free(mrows);
    free(mn);
    free(m1);
    free(h);
    free(qkv);
    free(wi);
    free(g);
    free(cg);
}

/* ================================================================== request handling (agent.py) */

static char *xstrdup_n(const char *s, size_t n)
{
    char *r = xmalloc(n + 1);
    memcpy(r, s, n);
    r[n] = 0;
    return r;
}

static char *replace_mask(const char *s, size_t n, const char *mask, size_t *outn)
{
    size_t ml = strlen(mask);
    Buf b = {0};
    for (size_t i = 0; i < n;) {
        if (i + ml <= n && !memcmp(s + i, mask, ml)) {
            buf_putc(&b, ' ');
            i += ml;
        } else {
            buf_putc(&b, s[i++]);
        }
    }
    if (!b.s) buf_puts(&b, "");
    *outn = b.n;
    return b.s;
}

static char *render_criterion(const JV *v)
{
    if (v->t == J_STR) return xstrdup_n(v->s, v->len);
    Buf b = {0};
    json_write(&b, v, 0);
    return b.s;
}

typedef struct {
    int t; /* 0 choice, 1 score, 2 noul */
    char *ins;
    int nopt;
    char **opt;
    const JV *crit;
} Q;

static const char *QNAME[3] = {"choice", "score", "noul"};

static char *fmt_opt(const char *prefix, const char *desc)
{
    Buf b = {0};
    buf_puts(&b, prefix);
    if (desc) {
        buf_puts(&b, ": ");
        buf_puts(&b, desc);
    }
    return b.s;
}

static int absent(const JV *v) { return !v || v->t == J_NULL || (v->t == J_STR && v->len == 0); }

/* Agent._to_internal + render_options */
static void q_parse(Q *q, const char *qid, const JV *qd)
{
    if (qd->t != J_OBJ) die("question '%s' must be an object", qid);
    const JV *ty = jget(qd, "type"), *ins = jget(qd, "instructions"), *crit = jget(qd, "criteria");
    if (!ty || ty->t != J_STR) die("question '%s': missing 'type'", qid);
    q->t = -1;
    for (int i = 0; i < 3; i++)
        if (!strcmp(ty->s, QNAME[i])) q->t = i;
    if (q->t < 0) die("question '%s': unknown type '%s' (choice|score|noul)", qid, ty->s);
    if (!ins) die("question '%s': missing 'instructions'", qid);
    if (ins->t == J_STR) q->ins = xstrdup_n(ins->s, ins->len);
    else {
        Buf b = {0};
        json_write(&b, ins, 1); /* json.dumps(ins) */
        q->ins = b.s;
    }
    q->crit = crit;
    if (q->t == 0) {
        if (!crit || (crit->t != J_OBJ && crit->t != J_ARR) || crit->n == 0) die("question '%s': choice needs non-empty 'criteria'", qid);
        q->nopt = (int)crit->n;
        q->opt = xmalloc(sizeof(char *) * (size_t)q->nopt);
        for (int i = 0; i < q->nopt; i++) {
            if (crit->t == J_ARR) {
                if (crit->v[i]->t != J_STR) die("question '%s': list criteria must be strings", qid);
                q->opt[i] = xstrdup_n(crit->v[i]->s, crit->v[i]->len);
            } else {
                const JV *v = crit->v[i];
                char *desc = absent(v) ? NULL : render_criterion(v);
                q->opt[i] = fmt_opt(crit->k[i], desc);
                free(desc);
            }
        }
    } else if (q->t == 1) {
        if (!crit || crit->t != J_ARR || crit->n == 0) die("question '%s': score needs a non-empty 'criteria' list", qid);
        q->nopt = (int)crit->n;
        q->opt = xmalloc(sizeof(char *) * (size_t)q->nopt);
        for (int i = 0; i < q->nopt; i++) {
            char pre[32];
            snprintf(pre, sizeof pre, "level %d", i);
            char *desc = render_criterion(crit->v[i]);
            q->opt[i] = fmt_opt(pre, desc);
            free(desc);
        }
    } else {
        if (crit && crit->t != J_OBJ && !absent(crit)) die("question '%s': noul 'criteria' must be an object", qid);
        const JV *fc = jget(crit, "false"), *tc = jget(crit, "true");
        q->nopt = 2;
        q->opt = xmalloc(sizeof(char *) * 2);
        char *fd = absent(fc) ? xstrdup_n("no, the statement does not hold", 31) : render_criterion(fc);
        char *td = absent(tc) ? xstrdup_n("yes, the statement holds", 24) : render_criterion(tc);
        q->opt[0] = fmt_opt("false", fd);
        q->opt[1] = fmt_opt("true", td);
        free(fd);
        free(td);
    }
}

static void q_free(Q *q)
{
    free(q->ins);
    for (int i = 0; i < q->nopt; i++) free(q->opt[i]);
    free(q->opt);
}

/* common.build_sequence: [CLS] <type> question: ins [SEP] [MASK] opt0 [MASK] opt1 ... [SEP] state [SEP] */
static void build_sequence(const Laya *M, const char *state, size_t slen, const Q *q, IVec *ids, IVec *markers)
{
    const Tok *T = &M->tok;
    size_t n;
    char *ins = replace_mask(q->ins, strlen(q->ins), T->mask_str, &n);
    Buf ht = {0};
    buf_puts(&ht, QNAME[q->t]);
    buf_puts(&ht, " question: ");
    buf_put(&ht, ins, n);
    free(ins);
    IVec head = {0};
    tok_encode(T, ht.s, ht.n, &head, M->head_max_len);
    free(ht.s);

    IVec *oi = xmalloc(sizeof(IVec) * (size_t)q->nopt);
    int total = 0;
    for (int i = 0; i < q->nopt; i++) {
        oi[i] = (IVec){0};
        size_t on;
        char *o = replace_mask(q->opt[i], strlen(q->opt[i]), T->mask_str, &on);
        Buf ob = {0};
        buf_putc(&ob, ' ');
        buf_put(&ob, o, on);
        free(o);
        iv_push(&oi[i], T->mask);
        IVec body = {0};
        tok_encode(T, ob.s, ob.n, &body, 48);
        for (int k = 0; k < body.n && k < 48; k++) iv_push(&oi[i], body.a[k]);
        free(body.a);
        free(ob.s);
        total += oi[i].n;
    }
    int opt_budget = M->head_max_len - total;
    if (opt_budget < 16) {
        int per = (M->head_max_len - 16) / (q->nopt > 1 ? q->nopt : 1);
        if (per < 4) per = 4;
        total = 0;
        for (int i = 0; i < q->nopt; i++) {
            if (oi[i].n > per) oi[i].n = per;
            total += oi[i].n;
        }
        opt_budget = M->head_max_len - total;
    }
    int take = opt_budget > 8 ? opt_budget : 8;
    if (head.n > take) head.n = take;

    iv_push(ids, T->cls);
    for (int i = 0; i < head.n; i++) iv_push(ids, head.a[i]);
    iv_push(ids, T->sep);
    for (int i = 0; i < q->nopt; i++) {
        iv_push(markers, ids->n);
        for (int k = 0; k < oi[i].n; k++) iv_push(ids, oi[i].a[k]);
        free(oi[i].a);
    }
    free(oi);
    free(head.a);
    iv_push(ids, T->sep);

    int room = M->max_len - ids->n - 1;
    if (room < 0) room = 0;
    size_t sn;
    char *st = replace_mask(state, slen, T->mask_str, &sn);
    IVec sv = {0};
    tok_encode(T, st, sn, &sv, room);
    free(st);
    for (int i = 0; i < sv.n && i < room; i++) iv_push(ids, sv.a[i]);
    free(sv.a);
    iv_push(ids, T->sep);
    if (ids->n > M->max_len) ids->n = M->max_len;
    int keep = 0;
    for (int i = 0; i < markers->n; i++)
        if (markers->a[i] < M->max_len) markers->a[keep++] = markers->a[i];
    markers->n = keep;
}

static void put_num(Buf *b, double x) /* round(x, 4), printed like Python's repr */
{
    char t[48];
    snprintf(t, sizeof t, "%.4f", x);
    char *dot = strchr(t, '.');
    size_t l = strlen(t);
    while (dot && l > 2 && t[l - 1] == '0' && t[l - 2] != '.') t[--l] = 0;
    if (!strcmp(t, "-0.0")) strcpy(t, "0.0");
    buf_puts(b, t);
}

static double temp_for(const Laya *M, int qt, int k)
{
    const char *sz = k <= 2 ? "2" : k <= 5 ? "3-5" : k <= 10 ? "6-10" : "11+";
    char key[32];
    snprintf(key, sizeof key, "%s:%s", QNAME[qt], sz);
    for (int i = 0; i < M->ntemps; i++)
        if (!strcmp(M->temps[i].key, key)) return M->temps[i].t;
    return M->temperature[qt];
}

char *laya_predict(Laya *M, const char *request_json)
{
    if (!M->have_model) die("model weights not loaded");
    int dbg = getenv("LAYA_DEBUG") != NULL;
    Arena A = {0};
    JV *root = jparse(&A, request_json, strlen(request_json));
    const JV *state = jget(root, "state"), *qs = jget(root, "questions");
    if (!state || !qs || qs->t != J_OBJ) die("request must be an object with 'state' and 'questions'");
    Buf sb = {0};
    if (state->t == J_STR) buf_put(&sb, state->s, state->len);
    else json_write(&sb, state, 0); /* serialize_state: json.dumps(state, ensure_ascii=False) */
    if (!sb.s) buf_puts(&sb, "");
    if (dbg) fprintf(stderr, "[state] %s\n", sb.s);

    Buf out = {0};
    buf_puts(&out, "{\"model\": \"laya-rl-agent\", \"answers\": {");
    long ntok = 0;
    for (size_t qi = 0; qi < qs->n; qi++) {
        const char *qid = qs->k[qi];
        Q q = {0};
        q_parse(&q, qid, qs->v[qi]);
        IVec ids = {0}, mk = {0};
        double t0 = now_ms();
        build_sequence(M, sb.s, sb.n, &q, &ids, &mk);
        if (mk.n != q.nopt) die("question '%s' options exceed head_max_len=%d", qid, M->head_max_len);
        if (dbg) {
            fprintf(stderr, "[%s] T=%d markers:", qid, ids.n);
            for (int i = 0; i < mk.n; i++) fprintf(stderr, " %d", mk.a[i]);
            fprintf(stderr, "\n[%s] ids:", qid);
            for (int i = 0; i < ids.n; i++) fprintf(stderr, " %d", ids.a[i]);
            fputc('\n', stderr);
        }
        ntok += ids.n;
        int k = q.nopt;
        float *logits = xmalloc(sizeof(float) * (size_t)k), actp;
        forward(M, ids.a, ids.n, mk.a, k, q.t, logits, &actp);
        if (dbg) {
            fprintf(stderr, "[%s] logits:", qid);
            for (int i = 0; i < k; i++) fprintf(stderr, " %.6f", logits[i]);
            fprintf(stderr, "\n[%s] act_prob(raw): %.6f   (%.0f ms)\n", qid, actp, now_ms() - t0);
        }

        /* calibration: temperature per (type, option count), softmax, entropy confidence */
        double ts = temp_for(M, q.t, k);
        if (ts < 1e-3) ts = 1e-3;
        double *p = xmalloc(sizeof(double) * (size_t)k), mx = -1e300, sum = 0;
        for (int i = 0; i < k; i++) {
            p[i] = logits[i] / ts;
            if (p[i] > mx) mx = p[i];
        }
        for (int i = 0; i < k; i++) {
            p[i] = exp(p[i] - mx);
            sum += p[i];
        }
        for (int i = 0; i < k; i++) p[i] /= sum;
        double conf = 1.0;
        if (k >= 2) {
            double ent = 0;
            for (int i = 0; i < k; i++) ent -= p[i] * log(p[i] < 1e-12 ? 1e-12 : p[i]);
            conf = 1.0 - ent / log((double)k);
            conf = conf < 0 ? 0 : conf > 1 ? 1 : conf;
        }
        /* round like the reference does, before deriving nothing else from the rounded values */
        if (qi) buf_puts(&out, ", ");
        json_str(&out, qid, strlen(qid), 0);
        buf_puts(&out, ": {\"type\": \"");
        buf_puts(&out, QNAME[q.t]);
        buf_puts(&out, "\", ");
        if (q.t == 0) {
            int am = 0;
            for (int i = 1; i < k; i++)
                if (p[i] > p[am]) am = i;
            buf_puts(&out, "\"choice\": ");
            const char *lab = q.crit->t == J_OBJ ? q.crit->k[am] : q.crit->v[am]->s;
            json_str(&out, lab, strlen(lab), 0);
            buf_puts(&out, ", \"probabilities\": {");
            for (int i = 0; i < k; i++) {
                if (i) buf_puts(&out, ", ");
                const char *l2 = q.crit->t == J_OBJ ? q.crit->k[i] : q.crit->v[i]->s;
                json_str(&out, l2, strlen(l2), 0);
                buf_puts(&out, ": ");
                put_num(&out, p[i]);
            }
            buf_puts(&out, "}, \"confidence\": ");
            put_num(&out, conf);
        } else if (q.t == 1) {
            double es = 0;
            for (int i = 0; i < k; i++) es += i * p[i];
            buf_puts(&out, "\"score\": ");
            put_num(&out, es);
            buf_puts(&out, ", \"legend\": {");
            for (int i = 0; i < k; i++) {
                char kb[16];
                snprintf(kb, sizeof kb, "\"%d\": ", i);
                if (i) buf_puts(&out, ", ");
                buf_puts(&out, kb);
                json_write(&out, q.crit->v[i], 0);
            }
            buf_puts(&out, "}, \"probabilities\": {");
            for (int i = 0; i < k; i++) {
                char kb[16];
                snprintf(kb, sizeof kb, "\"%d\": ", i);
                if (i) buf_puts(&out, ", ");
                buf_puts(&out, kb);
                put_num(&out, p[i]);
            }
            buf_puts(&out, "}, \"confidence\": ");
            put_num(&out, conf);
        } else {
            buf_puts(&out, "\"noul\": ");
            put_num(&out, p[1]);
            buf_puts(&out, ", \"confidence\": ");
            put_num(&out, p[1] > 1.0 - p[1] ? p[1] : 1.0 - p[1]);
        }
        buf_puts(&out, ", \"action\": {\"act_probability\": ");
        put_num(&out, actp);
        buf_puts(&out, "}}");
        free(p);
        free(logits);
        free(ids.a);
        free(mk.a);
        q_free(&q);
    }
    char tail[96];
    snprintf(tail, sizeof tail, "}, \"usage\": {\"input_tokens\": %ld, \"output_tokens\": 0}}", ntok);
    buf_puts(&out, tail);
    free(sb.s);
    arena_free(&A);
    return out.s;
}

/* ================================================================== HTTP server */

#define LAYA_DEFAULT_PORT 29417 /* below the Linux ephemeral range (32768+), unassigned by IANA */

/* Model + tokenizer are loaded once. Every accepted connection is handled concurrently with the others, but
 * inference itself is serialized so requests queue instead of oversubscribing the cores, and /status is answered
 * while an inference is running.
 *   POSIX:   one forked child per connection (weights shared copy-on-write, a crash only kills its child);
 *            serialization = fcntl record lock, released automatically if a child dies.
 *   Windows: one thread per connection; serialization = CRITICAL_SECTION.
 * In both cases a request that fails (die()) is unwound with longjmp and everything it allocated is released. */

typedef struct {
    long served, failed, in_flight;
    double last_ms, total_ms, load_ms, start_ms;
} Stats;
static Stats *g_st;
static int g_port;

#if defined(__GNUC__) || defined(__clang__)
#define ATOMIC_ADD(p, v) __atomic_add_fetch((p), (v), __ATOMIC_SEQ_CST)
#elif defined(_WIN32)
#define ATOMIC_ADD(p, v) (InterlockedExchangeAdd((volatile LONG *)(p), (LONG)(v)) + (v))
#else
#define ATOMIC_ADD(p, v) (*(p) += (v))
#endif

#ifdef _WIN32
typedef SOCKET sock_t;
#define SHUT_WR SD_SEND
static void sock_close(sock_t s) { closesocket(s); }
static long sock_recv(sock_t s, void *b, size_t n)
{
    int r = recv(s, (char *)b, (int)(n > (1u << 30) ? (1u << 30) : n), 0);
    if (r < 0) return WSAGetLastError() == WSAEINTR ? -2 : -1;
    return r;
}
static long sock_send(sock_t s, const void *b, size_t n)
{
    int r = send(s, (const char *)b, (int)(n > (1u << 30) ? (1u << 30) : n), 0);
    if (r < 0) return WSAGetLastError() == WSAEINTR ? -2 : -1;
    return r;
}
static void sock_timeouts(sock_t s, int sec)
{
    DWORD ms = (DWORD)sec * 1000;
    setsockopt(s, SOL_SOCKET, SO_RCVTIMEO, (const char *)&ms, sizeof ms);
    setsockopt(s, SOL_SOCKET, SO_SNDTIMEO, (const char *)&ms, sizeof ms);
}
static CRITICAL_SECTION g_cs;
static volatile LONG g_running;
static void inf_lock(void)
{
    EnterCriticalSection(&g_cs);
    g_running = 1;
}
static void inf_unlock(void)
{
    g_running = 0;
    LeaveCriticalSection(&g_cs);
}
static int inference_running(void) { return g_running != 0; }
#else
typedef int sock_t;
static void sock_close(sock_t s) { close(s); }
static long sock_recv(sock_t s, void *b, size_t n)
{
    ssize_t r = recv(s, b, n, 0);
    if (r < 0) return errno == EINTR ? -2 : -1;
    return (long)r;
}
static long sock_send(sock_t s, const void *b, size_t n)
{
    ssize_t r = send(s, b, n, 0);
    if (r < 0) return errno == EINTR ? -2 : -1;
    return (long)r;
}
static void sock_timeouts(sock_t s, int sec)
{
    struct timeval tv = {sec, 0};
    setsockopt(s, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof tv);
    setsockopt(s, SOL_SOCKET, SO_SNDTIMEO, &tv, sizeof tv);
}
static int g_lockfd = -1;
static void inf_lockop(int type)
{
    struct flock fl;
    memset(&fl, 0, sizeof fl);
    fl.l_type = (short)type;
    fl.l_whence = SEEK_SET;
    while (fcntl(g_lockfd, F_SETLKW, &fl) != 0 && errno == EINTR) {}
}
static void inf_lock(void) { inf_lockop(F_WRLCK); }
static void inf_unlock(void) { inf_lockop(F_UNLCK); }
static int inference_running(void)
{
    struct flock fl;
    memset(&fl, 0, sizeof fl);
    fl.l_type = F_WRLCK;
    fl.l_whence = SEEK_SET;
    if (fcntl(g_lockfd, F_GETLK, &fl) != 0) return -1;
    return fl.l_type != F_UNLCK;
}
#endif

static int send_all(sock_t fd, const char *p, size_t n)
{
    while (n) {
        long w = sock_send(fd, p, n);
        if (w == -2) continue;
        if (w <= 0) return -1;
        p += w;
        n -= (size_t)w;
    }
    return 0;
}

static void http_reply(sock_t fd, int code, const char *reason, const char *body, size_t n, double ms)
{
    char h[320];
    int k = snprintf(h, sizeof h,
                     "HTTP/1.1 %d %s\r\nContent-Type: application/json\r\nContent-Length: %zu\r\nConnection: close\r\n", code, reason, n);
    if (ms >= 0) k += snprintf(h + k, sizeof h - (size_t)k, "X-Laya-Ms: %.1f\r\n", ms);
    k += snprintf(h + k, sizeof h - (size_t)k, "\r\n");
    if (send_all(fd, h, (size_t)k) == 0) send_all(fd, body, n);
}

static void http_error(sock_t fd, int code, const char *reason, const char *msg)
{
    Buf b = {0};
    buf_puts(&b, "{\"error\": ");
    json_str(&b, msg, strlen(msg), 0);
    buf_putc(&b, '}');
    http_reply(fd, code, reason, b.s, b.n, -1);
    free(b.s);
}

/* case-insensitive helpers; `lower` must already be lower-case */
static int ci_prefix(const char *s, const char *lower)
{
    for (; *lower; s++, lower++)
        if (tolower((unsigned char)*s) != *lower) return 0;
    return 1;
}
static int ci_line_has(const char *s, const char *lower) /* within the current header line only */
{
    size_t n = strlen(lower);
    for (; *s && *s != '\r' && *s != '\n'; s++) {
        size_t i = 0;
        while (i < n && s[i] && tolower((unsigned char)s[i]) == lower[i]) i++;
        if (i == n) return 1;
    }
    return 0;
}

#define HTTP_MAX_HEADER (64 * 1024)
#define HTTP_MAX_BODY (32u * 1024 * 1024)

/* Reads one request. Returns 0 on success (method/path filled, *body malloc'd, NUL-terminated), -1 if the peer went
 * away, else an HTTP status to answer with. */
static int http_read(sock_t fd, char *method, char *path, char **body, size_t *blen)
{
    size_t cap = 8192, n = 0, hend = 0;
    char *b = xmalloc(cap + 1);
    for (;;) {
        if (n == cap) {
            if (cap >= HTTP_MAX_HEADER) {
                free(b);
                return 431;
            }
            cap *= 2;
            b = xrealloc(b, cap + 1);
        }
        long r = sock_recv(fd, b + n, cap - n);
        if (r == -2) continue;
        if (r <= 0) {
            free(b);
            return -1;
        }
        size_t from = n > 3 ? n - 3 : 0;
        n += (size_t)r;
        b[n] = 0;
        char *e = strstr(b + from, "\r\n\r\n");
        if (e) {
            hend = (size_t)(e - b) + 4;
            break;
        }
    }
    method[0] = path[0] = 0;
    if (sscanf(b, "%15s %1023s", method, path) != 2) {
        free(b);
        return 400;
    }
    char *q = strchr(path, '?');
    if (q) *q = 0;
    size_t cl = 0;
    int expect = 0;
    for (char *l = strstr(b, "\r\n"); l && l + 2 < b + hend; l = strstr(l + 2, "\r\n")) {
        char *h = l + 2;
        if (ci_prefix(h, "content-length:")) cl = (size_t)strtoull(h + 15, NULL, 10);
        else if (ci_prefix(h, "expect:") && ci_line_has(h, "100-continue")) expect = 1;
        else if (ci_prefix(h, "transfer-encoding:") && ci_line_has(h, "chunked")) {
            free(b);
            return 501;
        }
    }
    if (cl > HTTP_MAX_BODY) {
        free(b);
        return 413;
    }
    char *body_buf = xmalloc(cl + 1);
    size_t have = n - hend;
    if (have > cl) have = cl;
    memcpy(body_buf, b + hend, have);
    free(b);
    if (have < cl && expect) send_all(fd, "HTTP/1.1 100 Continue\r\n\r\n", 25);
    while (have < cl) {
        long r = sock_recv(fd, body_buf + have, cl - have);
        if (r == -2) continue;
        if (r <= 0) {
            free(body_buf);
            return -1;
        }
        have += (size_t)r;
    }
    body_buf[cl] = 0;
    *body = body_buf;
    *blen = cl;
    return 0;
}

static void status_json(Laya *M, Buf *b)
{
    double up = (now_ms() - g_st->start_ms) / 1e3;
    long served = g_st->served;
    char t[768];
#ifdef _OPENMP
    int threads = omp_get_max_threads();
#else
    int threads = 1;
#endif
    snprintf(t, sizeof t,
             "{\"status\": \"ready\", \"model\": \"laya-rl-agent\", \"port\": %d, \"uptime_s\": %.1f, \"load_ms\": %.0f, "
             "\"requests\": {\"served\": %ld, \"failed\": %ld, \"in_flight\": %ld, \"inference_running\": %s}, "
             "\"latency_ms\": {\"last\": %.1f, \"avg\": %.1f}, "
             "\"engine\": {\"hidden\": %d, \"layers\": %d, \"heads\": %d, \"vocab\": %d, \"max_len\": %d, \"head_max_len\": %d, "
             "\"threads\": %d, \"openmp\": %s}}",
             g_port, up, g_st->load_ms, served, g_st->failed, g_st->in_flight, inference_running() > 0 ? "true" : "false",
             g_st->last_ms, served ? g_st->total_ms / (double)served : 0.0, M->d, M->nl, M->nh, M->vocab, M->max_len,
             M->head_max_len, threads,
#ifdef _OPENMP
             "true"
#else
             "false"
#endif
    );
    buf_puts(b, t);
}

/* Handles one connection (forked child on POSIX, thread on Windows). Never returns via die(): errors become HTTP 400. */
static void serve_conn(Laya *M, sock_t fd)
{
    sock_timeouts(fd, 15);
    laya_jmp jb;
    g_jmp = &jb;
    volatile int is_predict = 0, locked = 0; /* read after longjmp */
    req_begin();
    if (LAYA_SETJMP(jb)) { /* die() was called somewhere below */
        if (locked) inf_unlock();
        req_abort();
        if (is_predict) ATOMIC_ADD(&g_st->failed, 1);
        http_error(fd, 400, "Bad Request", g_errmsg);
        g_jmp = NULL;
        return;
    }
    char method[16], path[1024];
    char *body = NULL;
    size_t blen = 0;
    int rc = http_read(fd, method, path, &body, &blen);
    if (rc < 0) {
        req_abort();
        g_jmp = NULL;
        return;
    }
    if (rc) {
        http_error(fd, rc, rc == 413 ? "Payload Too Large" : rc == 431 ? "Request Header Fields Too Large" : rc == 501 ? "Not Implemented" : "Bad Request",
                   rc == 501 ? "chunked request bodies are not supported; send Content-Length" : "malformed request");
    } else if (!strcmp(path, "/status") || !strcmp(path, "/health") || !strcmp(path, "/healthz")) {
        if (strcmp(method, "GET") && strcmp(method, "HEAD")) {
            http_error(fd, 405, "Method Not Allowed", "use GET");
        } else {
            Buf b = {0};
            status_json(M, &b);
            http_reply(fd, 200, "OK", b.s, b.n, -1);
            free(b.s);
        }
    } else if (!strcmp(path, "/predict") || !strcmp(path, "/questions") || !strcmp(path, "/")) {
        if (strcmp(method, "POST")) {
            http_error(fd, 405, "Method Not Allowed", "use POST with a JSON body {\"state\": ..., \"questions\": {...}}");
        } else {
            is_predict = 1;
            inf_lock();
            locked = 1;
            double t0 = now_ms();
            char *res = laya_predict(M, body);
            double ms = now_ms() - t0;
            g_st->last_ms = ms;
            g_st->total_ms += ms;
            ATOMIC_ADD(&g_st->served, 1);
            locked = 0;
            inf_unlock();
            http_reply(fd, 200, "OK", res, strlen(res), ms);
            free(res);
        }
    } else {
        http_error(fd, 404, "Not Found", "endpoints: GET /status, POST /predict");
    }
    free(body);
    req_end();
    g_jmp = NULL;
}

#ifdef _WIN32
typedef struct {
    Laya *M;
    SOCKET s;
} ConnArg;

static unsigned __stdcall conn_thread(void *p)
{
    ConnArg a = *(ConnArg *)p;
    free(p);
    serve_conn(a.M, a.s);
    shutdown(a.s, SHUT_WR);
    closesocket(a.s);
    ATOMIC_ADD(&g_st->in_flight, -1);
    return 0;
}
#else
static void on_sigchld(int sig)
{
    (void)sig;
    int saved = errno;
    while (waitpid(-1, NULL, WNOHANG) > 0) ATOMIC_ADD(&g_st->in_flight, -1);
    errno = saved;
}
#endif

static void serve_forever(Laya *M, const char *bind_addr, int port, double load_ms)
{
#ifdef _WIN32
    WSADATA wsa;
    if (WSAStartup(MAKEWORD(2, 2), &wsa) != 0) die("WSAStartup failed");
    g_st = xcalloc(1, sizeof *g_st);
    InitializeCriticalSection(&g_cs);
#else
    g_st = mmap(NULL, sizeof *g_st, PROT_READ | PROT_WRITE, MAP_SHARED | MAP_ANONYMOUS, -1, 0);
    if (g_st == MAP_FAILED) die("mmap failed");
    memset(g_st, 0, sizeof *g_st);
    FILE *tf = tmpfile();
    if (!tf) die("tmpfile failed");
    g_lockfd = fileno(tf);
    signal(SIGPIPE, SIG_IGN);
    struct sigaction sa;
    memset(&sa, 0, sizeof sa);
    sa.sa_handler = on_sigchld;
    sa.sa_flags = SA_RESTART | SA_NOCLDSTOP;
    sigaction(SIGCHLD, &sa, NULL);
#endif
    g_st->load_ms = load_ms;
    g_st->start_ms = now_ms();
    g_port = port;

    sock_t ls = socket(AF_INET, SOCK_STREAM, 0);
#ifdef _WIN32
    if (ls == INVALID_SOCKET) die("socket failed");
    int one = 1;
    setsockopt(ls, SOL_SOCKET, SO_EXCLUSIVEADDRUSE, (const char *)&one, sizeof one); /* SO_REUSEADDR would allow port hijacking here */
#else
    if (ls < 0) die("socket failed");
    int one = 1;
    setsockopt(ls, SOL_SOCKET, SO_REUSEADDR, &one, sizeof one);
#endif
    struct sockaddr_in a;
    memset(&a, 0, sizeof a);
    a.sin_family = AF_INET;
    a.sin_port = htons((uint16_t)port);
    if (inet_pton(AF_INET, bind_addr, &a.sin_addr) != 1) die("bad bind address '%s'", bind_addr);
    if (bind(ls, (struct sockaddr *)&a, sizeof a) != 0) die("cannot bind %s:%d (port in use? try --serve OTHER_PORT)", bind_addr, port);
    if (listen(ls, 128) != 0) die("listen failed");
    fprintf(stderr, "laya: model loaded in %.0f ms; listening on http://%s:%d  (GET /status, POST /predict)\n", load_ms, bind_addr, port);
    fflush(stderr);
    for (;;) {
        sock_t c = accept(ls, NULL, NULL);
#ifdef _WIN32
        if (c == INVALID_SOCKET) {
            Sleep(10);
            continue;
        }
#else
        if (c < 0) {
            if (errno != EINTR) perror("laya: accept");
            continue;
        }
#endif
        if (g_st->in_flight >= 64) {
            http_error(c, 503, "Service Unavailable", "too many connections in flight");
            sock_close(c);
            continue;
        }
        ATOMIC_ADD(&g_st->in_flight, 1);
#ifdef _WIN32
        /* NB: must come from xmalloc -- free() is xfree(), which expects the allocator's header (a raw malloc'd
         * block here corrupted the Windows heap: STATUS_HEAP_CORRUPTION 0xc0000374). */
        ConnArg *ca = xmalloc(sizeof *ca);
        ca->M = M;
        ca->s = c;
        HANDLE th = (HANDLE)_beginthreadex(NULL, 0, conn_thread, ca, 0, NULL);
        if (!th) {
            ATOMIC_ADD(&g_st->in_flight, -1);
            http_error(c, 503, "Service Unavailable", "cannot start worker thread");
            sock_close(c);
            free(ca);
        } else {
            CloseHandle(th);
        }
#else
        pid_t pid = fork();
        if (pid == 0) {
            sock_close(ls);
            serve_conn(M, c);
            shutdown(c, SHUT_WR);
            sock_close(c);
            _exit(0);
        }
        if (pid < 0) {
            ATOMIC_ADD(&g_st->in_flight, -1);
            http_error(c, 503, "Service Unavailable", "fork failed");
        }
        sock_close(c);
#endif
    }
}

/* ================================================================== CLI */

#ifndef LAYA_NO_MAIN
int main(int argc, char **argv)
{
    if (argc < 3) {
        fprintf(stderr,
                "usage: %s MODEL_DIR REQUEST.json|-\n"
                "       %s MODEL_DIR --tokenize TEXT\n"
                "       %s MODEL_DIR --list-tensors\n"
                "       %s MODEL_DIR --serve [PORT] [--bind ADDR]   (HTTP server, default 127.0.0.1:%d)\n",
                argv[0], argv[0], argv[0], argv[0], LAYA_DEFAULT_PORT);
        return 2;
    }
    if (!strcmp(argv[2], "--tokenize")) {
        if (argc < 4) die("--tokenize needs a text argument");
        Laya *M = laya_load_impl(argv[1], 1, 0);
        IVec v = {0};
        tok_encode(&M->tok, argv[3], strlen(argv[3]), &v, -1);
        for (int i = 0; i < v.n; i++) printf(i ? " %d" : "%d", v.a[i]);
        putchar('\n');
        free(v.a);
        laya_free(M);
        return 0;
    }
    if (!strcmp(argv[2], "--list-tensors")) {
        laya_load_impl(argv[1], 0, 1);
        return 0;
    }
    if (!strcmp(argv[2], "--serve")) {
        int port = LAYA_DEFAULT_PORT;
        const char *bind_addr = "127.0.0.1";
        for (int i = 3; i < argc; i++) {
            if (!strcmp(argv[i], "--bind") && i + 1 < argc) bind_addr = argv[++i];
            else if (atoi(argv[i]) > 0 && atoi(argv[i]) < 65536) port = atoi(argv[i]);
            else die("unknown argument '%s' (usage: MODEL_DIR --serve [PORT] [--bind ADDR])", argv[i]);
        }
        double ts = now_ms();
        Laya *M = laya_load(argv[1]);
        serve_forever(M, bind_addr, port, now_ms() - ts);
        return 0;
    }
    double t0 = now_ms();
    Laya *M = laya_load(argv[1]);
    if (getenv("LAYA_DEBUG")) fprintf(stderr, "[load] %.0f ms\n", now_ms() - t0);
    char *req = read_file(argv[2], NULL);
    char *res = laya_predict(M, req);
    puts(res);
    free(res);
    free(req);
    laya_free(M);
    return 0;
}
#endif
