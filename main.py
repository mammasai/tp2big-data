"""TP02 - Big Data: OOM vs Pandas chunking vs Dask."""
import os, sys, time, json, threading
import pandas as pd
import psutil
import matplotlib.pyplot as plt

FILE = "train.csv"
CHUNK_SIZE = 100_000
RUN_TASK1 = True          # False = skip Task 1, reuse saved results
KILL_PCT = 92             # safety: stop if system RAM >= this %
SHOW_PLOTS = False        # True = open plot windows
RESULTS = {}              # {method: {time, peak_mb, peak_pct, status, timeline}}


# ---------- Memory monitor ----------
class MemoryMonitor:
    def __init__(self, name, kill_pct=None, step=0.2):
        self.name, self.kill_pct, self.step = name, kill_pct, step
        self.peak_mb = self.peak_pct = 0.0
        self.timeline = []                       # (seconds, MB)
        self._stop = threading.Event()
        self._p = psutil.Process(os.getpid())

    def _rss_mb(self):                           # process + children
        t = self._p.memory_info().rss
        for c in self._p.children(recursive=True):
            try: t += c.memory_info().rss
            except psutil.NoSuchProcess: pass
        return t / 1024**2

    def _run(self):
        t0 = time.time()
        while not self._stop.is_set():
            mb, pct = self._rss_mb(), psutil.virtual_memory().percent
            self.peak_mb, self.peak_pct = max(self.peak_mb, mb), max(self.peak_pct, pct)
            self.timeline.append((time.time() - t0, mb))
            if self.kill_pct and pct >= self.kill_pct:
                print(f"[!] RAM {pct:.0f}% -> abort (OOM)")
                RESULTS[self.name] = self.result(None, "OOM")
                save()
                os._exit(1)
            time.sleep(self.step)

    def result(self, elapsed, status="OK"):
        return {"time": elapsed, "peak_mb": self.peak_mb,
                "peak_pct": self.peak_pct, "status": status,
                "timeline": self.timeline}

    def __enter__(self):
        self._t = threading.Thread(target=self._run, daemon=True)
        self._t.start()
        return self

    def __exit__(self, *a):
        self._stop.set(); self._t.join()


def save():
    json.dump(RESULTS, open("results.json", "w", encoding="utf-8"))


# ---------- Task 1: standard pandas ----------
def task1():
    print("\n== Task 1: pd.read_csv ==")
    start, status = time.time(), "OK"
    with MemoryMonitor("Standard Pandas", KILL_PCT) as m:
        try:
            df = pd.read_csv(FILE)
            print("Loaded:", df.shape)
            del df
        except MemoryError as e:
            status = "OOM"
            print("MemoryError:", e)
    RESULTS["Standard Pandas"] = m.result(
        None if status == "OOM" else time.time() - start, status)
    print(f"Peak: {m.peak_mb:.0f} MB | System: {m.peak_pct:.0f}%")
    save()


# ---------- Task 2: pandas chunking ----------
def task2():
    print(f"\n== Task 2: chunking ({CHUNK_SIZE:,} rows) ==")
    cols = pd.read_csv(FILE, nrows=1000).select_dtypes("number").columns.tolist()
    print("Numeric cols:", cols)
    s = pd.Series(0.0, index=cols)               # running sum
    n = pd.Series(0, index=cols)                 # running count
    lo = pd.Series(float("inf"), index=cols)
    hi = pd.Series(float("-inf"), index=cols)
    rows = k = 0
    start = time.time()
    with MemoryMonitor("Pandas Chunking") as m:
        for ch in pd.read_csv(FILE, chunksize=CHUNK_SIZE, usecols=cols):
            s += ch.sum(); n += ch.count()
            lo = pd.concat([lo, ch.min()], axis=1).min(axis=1)
            hi = pd.concat([hi, ch.max()], axis=1).max(axis=1)
            rows += len(ch); k += 1
            if k % 50 == 0: print(f"  chunk {k} | rows {rows:,}")
    el = time.time() - start
    out = pd.DataFrame({"sum": s, "mean": s / n, "min": lo, "max": hi})
    print(f"Rows: {rows:,} | Chunks: {k}\n{out}")
    print(f"Time: {el:.1f}s | Peak: {m.peak_mb:.0f} MB")
    RESULTS["Pandas Chunking"] = m.result(el)
    save()
    out.to_csv("summary_pandas.csv")
    return cols


# ---------- Task 3: Dask ----------
def draw_pipeline():
    """Fallback image of the lazy pipeline (no Graphviz needed)."""
    fig, ax = plt.subplots(figsize=(11, 4)); ax.axis("off")
    rows = [("Unoptimized", 0.72, "#C00000",
             ["read_csv\n(all columns)", "partitions\n(64MB)", "partial\nmean/sum/min/max", "aggregate", "compute()"]),
            ("Optimized", 0.22, "#548235",
             ["read_csv\n(numeric cols only)", "partitions\n(64MB)", "partial\nmean/sum/min/max", "aggregate\n(one shared pass)", "compute()"])]
    for label, y, col, steps in rows:
        ax.text(0.0, y + 0.1, label, fontsize=12, weight="bold", color=col)
        for i, t in enumerate(steps):
            x = 0.02 + i * 0.2
            ax.text(x, y - 0.05, t, ha="left", va="center", fontsize=9,
                    bbox=dict(boxstyle="round", fc="white", ec=col))
            if i < len(steps) - 1:
                ax.annotate("", xy=(x + 0.18, y - 0.05), xytext=(x + 0.145, y - 0.05),
                            arrowprops=dict(arrowstyle="->", color=col))
    ax.set_title("Dask lazy pipeline (task graph overview)")
    plt.savefig("dask_pipeline.png", dpi=150, bbox_inches="tight"); plt.close(fig)


def task3(cols):
    import dask
    import dask.dataframe as dd
    from dask.diagnostics import ProgressBar
    print("\n== Task 3: Dask ==")

    # Lazy: only builds the plan, reads nothing
    t0 = time.time()
    raw = dd.read_csv(FILE, blocksize="64MB", assume_missing=True)
    print(f"Partitions: {raw.npartitions} | Plan time: {time.time()-t0:.3f}s (lazy)")

    # Optimization: read only needed columns
    ddf = dd.read_csv(FILE, blocksize="64MB", assume_missing=True, usecols=cols)
    mean, tot, mn, mx, cnt = ddf.mean(), ddf.sum(), ddf.min(), ddf.max(), ddf.shape[0]
    print("Lazy object:", type(mean).__name__, "(no value yet)")

    # Graph inspection (optional: never stops the program)
    draw_pipeline()
    print("Saved: dask_pipeline.png")
    try:
        print("Expression tree (optimized):")
        mean.optimize().pprint()
    except Exception as e:
        print("pprint skipped:", type(e).__name__)
    try:
        mean.visualize("dask_graph_optimized.png")
        print("Saved: dask_graph_optimized.png")
    except Exception as e:
        print("Graphviz image skipped:", type(e).__name__)

    # Single compute = file read once, threads in parallel
    start = time.time()
    with MemoryMonitor("Dask DataFrame") as m, ProgressBar():
        a, b, c, d, e = dask.compute(mean, tot, mn, mx, cnt, scheduler="threads")
    el = time.time() - start
    out = pd.DataFrame({"sum": b, "mean": a, "min": c, "max": d})
    print(f"Rows: {e:,}\n{out}")
    print(f"Time: {el:.1f}s | Peak: {m.peak_mb:.0f} MB")
    RESULTS["Dask DataFrame"] = m.result(el)
    save()
    out.to_csv("summary_dask.csv")


# ---------- Report + plots ----------
def report():
    print("\n== Benchmark ==")
    df = pd.DataFrame([{
        "Method": k,
        "Status": v["status"],
        "Time (s)": "N/A" if v["time"] is None else round(v["time"], 1),
        "Peak RAM (MB)": round(v["peak_mb"]),
        "System RAM (%)": round(v["peak_pct"], 1)} for k, v in RESULTS.items()])
    print(df.to_string(index=False))
    df.to_csv("benchmark.csv", index=False)

    names = list(RESULTS)
    colors = ["#C00000", "#2E75B6", "#548235"][:len(names)]

    # Plot 1: time + peak memory
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    times = [RESULTS[k]["time"] or 0 for k in names]
    ax[0].bar(names, times, color=colors)
    ax[0].set_title("Execution Time (s)")
    for i, k in enumerate(names):                # label OOM bars
        if RESULTS[k]["time"] is None:
            ax[0].text(i, 0, "OOM", ha="center", va="bottom", color="red")
    ax[1].bar(names, [RESULTS[k]["peak_mb"] for k in names], color=colors)
    ax[1].set_title("Peak Memory (MB)")
    for a in ax: a.tick_params(axis="x", rotation=15)
    plt.tight_layout(); plt.savefig("benchmark.png", dpi=150)

    # Plot 2: memory over time
    plt.figure(figsize=(9, 4))
    for k, c in zip(names, colors):
        tl = RESULTS[k]["timeline"]
        if tl:
            plt.plot([t for t, _ in tl], [m for _, m in tl], label=k, color=c)
    plt.xlabel("Time (s)"); plt.ylabel("Memory (MB)")
    plt.title("Memory Usage Over Time"); plt.legend(); plt.grid(alpha=.3)
    plt.tight_layout(); plt.savefig("memory_timeline.png", dpi=150)

    print("Saved: benchmark.csv, benchmark.png, memory_timeline.png")
    if SHOW_PLOTS: plt.show()


if __name__ == "__main__":
    if not os.path.exists(FILE):
        sys.exit(f"{FILE} not found. Unzip train.csv.zip first.")
    if "--task1" in sys.argv:                    # child process: Task 1 only
        task1()
        sys.exit(0)
    print(f"File: {os.path.getsize(FILE)/1024**3:.2f} GB | "
          f"RAM: {psutil.virtual_memory().total/1024**3:.1f} GB | "
          f"CPU: {psutil.cpu_count()}")
    if RUN_TASK1:                                # isolated: crash won't stop main
        import subprocess
        subprocess.run([sys.executable, __file__, "--task1"])
    if os.path.exists("results.json"):           # load Task 1 result
        RESULTS.update(json.load(open("results.json", encoding="utf-8")))
    cols = task2()
    task3(cols)
    report()