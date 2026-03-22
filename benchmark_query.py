import argparse
import json
import statistics
import time
import urllib.error
import urllib.request


DEFAULT_QUERIES = [
    "Что такое Метеор-М?",
    "Какое энергопитание у Метеора-М?",
    "Сколько солнечных панелей у Метеора-М?",
]


def post_json(url: str, payload: dict) -> tuple[dict, float]:
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    started = time.perf_counter()
    with urllib.request.urlopen(request, timeout=120) as response:
        body = response.read().decode("utf-8")
    elapsed = time.perf_counter() - started
    return json.loads(body), elapsed


def wait_for_server(base_url: str, ready_path: str, wait_seconds: float, poll_interval: float) -> bool:
    ready_url = base_url.rstrip("/") + "/" + ready_path.lstrip("/")
    deadline = time.time() + wait_seconds

    while time.time() < deadline:
        try:
            with urllib.request.urlopen(ready_url, timeout=10) as response:
                if 200 <= response.status < 500:
                    return True
        except Exception:
            pass
        time.sleep(poll_interval)
    return False


def format_metric(values: list[float]) -> str:
    if not values:
        return "n/a"
    return f"{statistics.mean(values):.3f}s avg | min {min(values):.3f}s | max {max(values):.3f}s"


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark /query timings on the running backend.")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000", help="Backend base URL.")
    parser.add_argument("--endpoint", default="/query", help="Endpoint to benchmark.")
    parser.add_argument("--repeat", type=int, default=3, help="How many times to run each query.")
    parser.add_argument("--ready-path", default="/docs", help="Path used to probe server readiness.")
    parser.add_argument("--wait-seconds", type=float, default=180.0, help="How long to wait for server startup.")
    parser.add_argument("--poll-interval", type=float, default=2.0, help="Readiness probe interval in seconds.")
    parser.add_argument(
        "--query",
        action="append",
        dest="queries",
        help="Query to benchmark. Can be provided multiple times.",
    )
    args = parser.parse_args()

    url = args.base_url.rstrip("/") + "/" + args.endpoint.lstrip("/")
    queries = args.queries or DEFAULT_QUERIES

    print(f"Waiting for server: {args.base_url.rstrip('/')}/{args.ready_path.lstrip('/')}")
    if not wait_for_server(args.base_url, args.ready_path, args.wait_seconds, args.poll_interval):
        print(
            "Server did not become ready in time. "
            "Check server.log or verify that port 8000 is listening before running the benchmark."
        )
        return 1

    client_totals: list[float] = []
    server_totals: list[float] = []
    retrieve_times: list[float] = []
    generate_times: list[float] = []

    print(f"Benchmarking {url}")
    print(f"Queries: {len(queries)} | Repeat: {args.repeat}")

    for query in queries:
        print(f"\n=== {query}")
        for attempt in range(1, args.repeat + 1):
            try:
                result, client_elapsed = post_json(url, {"text": query})
            except urllib.error.HTTPError as exc:
                print(f"[{attempt}] HTTP {exc.code}: {exc.read().decode('utf-8', errors='replace')}")
                return 1
            except Exception as exc:
                print(f"[{attempt}] Request failed: {exc}")
                return 1

            timing = result.get("timing") or {}
            server_total = timing.get("total")
            retrieve = timing.get("retrieve")
            generate = timing.get("generate")

            client_totals.append(client_elapsed)
            if isinstance(server_total, (int, float)):
                server_totals.append(float(server_total))
            if isinstance(retrieve, (int, float)):
                retrieve_times.append(float(retrieve))
            if isinstance(generate, (int, float)):
                generate_times.append(float(generate))

            answer = (result.get("answer") or "").strip()
            print(
                f"[{attempt}] client={client_elapsed:.3f}s "
                f"server={server_total if server_total is not None else 'n/a'} "
                f"retrieve={retrieve if retrieve is not None else 'n/a'} "
                f"generate={generate if generate is not None else 'n/a'}"
            )
            print(f"answer: {answer}")

    print("\n=== Summary")
    print(f"client_total: {format_metric(client_totals)}")
    print(f"server_total: {format_metric(server_totals)}")
    print(f"retrieve: {format_metric(retrieve_times)}")
    print(f"generate: {format_metric(generate_times)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
