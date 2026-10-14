"""
scripts/simulate_ai_engine.py
================================
MuleGuard.AI — Enterprise AI Engine Simulator & Benchmark Suite

Standalone enterprise test bench for the core ML scoring service:
  POST http://localhost:8002/score
  GET  http://localhost:8002/health

Usage Examples:
  # Check service health & loaded model artifacts
  python scripts/simulate_ai_engine.py --health

  # Run a specific enterprise scenario
  python scripts/simulate_ai_engine.py --scenario mule

  # Run all 8 enterprise scenarios and print executive comparison table
  python scripts/simulate_ai_engine.py --scenario all

  # Run batch stress benchmark (50 transactions) to measure p50/p95/p99 latency
  python scripts/simulate_ai_engine.py --stress --count 50

  # Interactive CLI transaction builder
  python scripts/simulate_ai_engine.py --interactive
"""

import argparse
import json
import os
import sys
import time
import urllib.request
import urllib.error
from datetime import datetime
from typing import Any, Dict, List, Optional

# Ensure UTF-8 output on Windows terminals
if sys.platform.startswith("win"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

# ANSI Color Codes for Enterprise Terminal Formatting
class Colors:
    HEADER = "\033[95m"
    BLUE = "\033[94m"
    CYAN = "\033[96m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    RESET = "\033[0m"

# Default AI Engine endpoint
DEFAULT_ENDPOINT = os.getenv("AI_ENGINE_URL", "http://localhost:8002")

# Enterprise Scenario Catalog
SCENARIOS = {
    "mule": {
        "name": "Mule Funnel Burst",
        "description": "High velocity outbound burst from known aggregator account (Redis hit).",
        "payload": {
            "transaction_id": "TXN-CLI-MULE-001",
            "from_account": "842E23DC0",
            "to_account": "ACC-MULE-LAYER-01",
            "amount": 48500.0,
            "payment_format": "Wire",
            "currency": "USD",
        },
    },
    "smurf": {
        "name": "Structuring / Smurfing",
        "description": "Calculated transaction ($9,950) right below FinCEN $10k reporting limit.",
        "payload": {
            "transaction_id": "TXN-CLI-SMURF-002",
            "from_account": "805401FF0",
            "to_account": "ACC-SHELL-CORP",
            "amount": 9950.0,
            "payment_format": "Cash",
            "currency": "USD",
        },
    },
    "crypto": {
        "name": "Crypto Layering & Cash-Out",
        "description": "Off-ramp cashout flow utilizing cryptocurrency rails.",
        "payload": {
            "transaction_id": "TXN-CLI-CRYPTO-003",
            "from_account": "83E6DBB70",
            "to_account": "EXCHANGE-HOT-WALLET-99",
            "amount": 27500.0,
            "payment_format": "Bitcoin",
            "currency": "USD",
        },
    },
    "velocity": {
        "name": "High-Velocity Burst",
        "description": "Sudden velocity spike on previously dormant consumer account.",
        "payload": {
            "transaction_id": "TXN-CLI-VELOCITY-004",
            "from_account": "817412030",
            "to_account": "ACC-BENEFICIARY-44",
            "amount": 14200.0,
            "payment_format": "Wire",
            "currency": "USD",
        },
    },
    "payroll": {
        "name": "Corporate Payroll",
        "description": "Legitimate established commercial corporate payroll distribution via ACH.",
        "payload": {
            "transaction_id": "TXN-CLI-PAYROLL-005",
            "from_account": "CORP-PAYROLL-MASTER",
            "to_account": "EMPLOYEE-DIR-DEP-042",
            "amount": 85400.0,
            "payment_format": "ACH",
            "currency": "USD",
        },
    },
    "retail": {
        "name": "Retail POS Micro-Txn",
        "description": "Routine consumer card-present retail debit transaction ($4.75).",
        "payload": {
            "transaction_id": "TXN-CLI-RETAIL-006",
            "from_account": "CONSUMER-CHECKING-99",
            "to_account": "MERCHANT-CAFE-POS-01",
            "amount": 4.75,
            "payment_format": "Debit Card",
            "currency": "USD",
        },
    },
    "coldstart": {
        "name": "Cold-Start Account",
        "description": "First transaction on new account. Verifies Autoencoder V6 Cold-Start Guard.",
        "payload": {
            "transaction_id": "TXN-CLI-COLDSTART-007",
            "from_account": "NEW-ACCT-FIRST-TXN",
            "to_account": "MERCHANT-ONLINE-SHOP",
            "amount": 120.0,
            "payment_format": "Credit Card",
            "currency": "USD",
        },
    },
    "crossborder": {
        "name": "Cross-Border Shell Wire",
        "description": "Inter-jurisdictional high-value wire between offshore shell entities.",
        "payload": {
            "transaction_id": "TXN-CLI-CROSSBORDER-008",
            "from_account": "80AF6F500",
            "to_account": "OVERSEAS-HOLDING-LTD",
            "amount": 62000.0,
            "payment_format": "Wire",
            "currency": "USD",
        },
    },
}


def print_banner():
    banner = f"""
{Colors.CYAN}{Colors.BOLD}========================================================================================
       M U L E G U A R D . A I   --   A I   E N G I N E   S I M U L A T O R
             Enterprise Standalone ML Scoring & Benchmarking Suite (:8002)
========================================================================================{Colors.RESET}"""
    print(banner)


def check_health(endpoint: str) -> Optional[Dict[str, Any]]:
    """Pings AI Engine /health endpoint."""
    url = f"{endpoint}/health"
    t0 = time.time()
    try:
        req = urllib.request.Request(url, headers={"Accept": "application/json"})
        with urllib.request.urlopen(req, timeout=3) as resp:
            data = json.loads(resp.read().decode())
            data["ping_ms"] = round((time.time() - t0) * 1000, 1)
            return data
    except Exception as exc:
        print(f"{Colors.RED}[!] Failed to connect to {url}: {exc}{Colors.RESET}")
        return None


def display_health(endpoint: str):
    print(f"\n{Colors.BOLD}Checking AI Engine Service Health at {endpoint}...{Colors.RESET}\n")
    health = check_health(endpoint)
    if not health:
        print(f"{Colors.RED}{Colors.BOLD}STATUS: SERVICE OFFLINE{Colors.RESET}")
        print("Please ensure the AI Engine container or local server is running on port 8002.")
        return

    status = health.get("status", "unknown")
    status_color = Colors.GREEN if status == "ok" else Colors.YELLOW
    print(f"  Service Status          : {status_color}{status.upper()}{Colors.RESET}")
    print(f"  LightGBM Model Loaded   : {'[PASS] YES' if health.get('lgbm_model_loaded') else '[FAIL] NO'}")
    print(f"  Autoencoder V6 Loaded   : {'[PASS] YES' if health.get('autoencoder_model_loaded') else '[FAIL] NO'}")
    print(f"  SHAP Explainer Ready    : {'[PASS] YES' if health.get('shap_explainer_ready') else '[FAIL] NO'}")
    print(f"  Redis Feature Store     : {'[PASS] CONNECTED' if health.get('redis_connected') else '[FAIL] DISCONNECTED'}")
    print(f"  Uptime                  : {health.get('uptime_seconds', 0):.1f} seconds")
    print(f"  Round-trip Ping Latency : {health.get('ping_ms', 0):.1f} ms")
    print("-" * 80)


def score_transaction(endpoint: str, payload: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Sends POST /score to AI Engine."""
    if "timestamp" not in payload:
        payload["timestamp"] = datetime.utcnow().isoformat() + "Z"

    url = f"{endpoint}/score"
    data_bytes = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data_bytes,
        headers={"Content-Type": "application/json", "Accept": "application/json"},
    )

    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=5) as resp:
            res = json.loads(resp.read().decode())
            res["network_rtt_ms"] = round((time.time() - t0) * 1000, 2)
            return res
    except urllib.error.HTTPError as err:
        body = err.read().decode()
        print(f"{Colors.RED}[HTTP {err.code}] Error: {body}{Colors.RESET}")
        return None
    except Exception as exc:
        print(f"{Colors.RED}[!] Scoring error: {exc}{Colors.RESET}")
        return None


def format_decision(decision: str) -> str:
    if decision == "BLOCK":
        return f"{Colors.RED}{Colors.BOLD}BLOCK{Colors.RESET}"
    elif decision == "REVIEW":
        return f"{Colors.YELLOW}{Colors.BOLD}REVIEW{Colors.RESET}"
    else:
        return f"{Colors.GREEN}{Colors.BOLD}APPROVE{Colors.RESET}"


def run_scenario(endpoint: str, key: str, export_file: Optional[str] = None):
    scenario = SCENARIOS.get(key)
    if not scenario:
        print(f"{Colors.RED}Unknown scenario '{key}'. Available: {', '.join(SCENARIOS.keys())}{Colors.RESET}")
        return

    print(f"\n{Colors.BOLD}>>> Executing Scenario: {Colors.CYAN}{scenario['name']}{Colors.RESET}")
    print(f"    {Colors.DIM}{scenario['description']}{Colors.RESET}\n")

    payload = dict(scenario["payload"])
    payload["timestamp"] = datetime.utcnow().isoformat() + "Z"

    print(f"    {Colors.BOLD}Payload:{Colors.RESET} From={payload['from_account']} -> To={payload['to_account']} | Amt=${payload['amount']:,.2f} | Format={payload['payment_format']}")

    res = score_transaction(endpoint, payload)
    if not res:
        return

    lgbm = res.get("lgbm_score", 0.0)
    ae = res.get("ae_score", 0.0)
    recon = res.get("ae_reconstruction_error", 0.0)
    composite = 0.7 * lgbm + 0.3 * ae
    dec = res.get("decision_preliminary", "APPROVE")
    source = res.get("feature_source", "unknown")
    latency = res.get("inference_latency_ms", 0.0)

    print(f"\n  {Colors.BOLD}--- SCORING RESULTS ---{Colors.RESET}")
    print(f"  Decision Preliminary   : {format_decision(dec)}")
    print(f"  Composite Risk Score   : {Colors.BOLD}{composite * 100:.2f}%{Colors.RESET}  (0.70*LGBM + 0.30*AE)")
    print(f"  LightGBM AML Prob      : {lgbm * 100:.3f}%")
    print(f"  Autoencoder Anomaly    : {ae * 100:.2f}%  (Recon MSE: {recon:.2f})")
    print(f"  Feature Store Source   : {Colors.GREEN if source == 'redis' else Colors.YELLOW}{source.upper()}{Colors.RESET}")
    print(f"  Inference Latency      : {latency:.1f} ms  (Target < 200ms: {'[PASS] MET' if latency <= 200 else '[WARN] BREACH'})")

    shap = res.get("shap_top_5", [])
    if shap:
        print(f"\n  {Colors.BOLD}--- TOP 5 SHAP ATTRIBUTIONS ---{Colors.RESET}")
        for i, item in enumerate(shap, 1):
            feat = item.get("feature", "unknown")
            contrib = item.get("shap_contribution", 0.0)
            direction = item.get("direction", "")
            color = Colors.RED if direction == "RAISES_RISK" else Colors.GREEN
            sign = "+" if contrib > 0 else ""
            print(f"    {i}. {feat:<32} : {color}{sign}{contrib:.4f}{Colors.RESET} ({direction})")

    print("-" * 80)

    if export_file:
        with open(export_file, "w") as f:
            json.dump({"scenario": key, "request": payload, "response": res}, f, indent=2)
        print(f"Result exported to {export_file}")


def run_all_scenarios(endpoint: str):
    print(f"\n{Colors.BOLD}>>> Running Executive Scorecard for All 8 Scenarios...{Colors.RESET}\n")
    print(f"{'#':<3} {'Scenario Name':<28} {'Amount':<12} {'Format':<10} {'LGBM':<9} {'AE':<8} {'Score':<8} {'Decision':<10} {'Source':<10} {'Latency'}")
    print("=" * 110)

    for i, (k, s) in enumerate(SCENARIOS.items(), 1):
        payload = dict(s["payload"])
        res = score_transaction(endpoint, payload)
        if not res:
            continue

        lgbm = res.get("lgbm_score", 0.0)
        ae = res.get("ae_score", 0.0)
        comp = 0.7 * lgbm + 0.3 * ae
        dec = res.get("decision_preliminary", "APPROVE")
        source = res.get("feature_source", "unknown")
        latency = res.get("inference_latency_ms", 0.0)

        dec_str = "BLOCK" if dec == "BLOCK" else "REVIEW" if dec == "REVIEW" else "APPROVE"
        color = Colors.RED if dec == "BLOCK" else Colors.YELLOW if dec == "REVIEW" else Colors.GREEN

        print(
            f"{i:<3} {s['name']:<28} ${payload['amount']:<11,.0f} {payload['payment_format']:<10} "
            f"{lgbm*100:<8.1f}% {ae*100:<7.1f}% {comp*100:<7.1f}% {color}{dec_str:<10}{Colors.RESET} "
            f"{source:<10} {latency:.1f}ms"
        )

    print("=" * 110)


def run_stress_benchmark(endpoint: str, count: int = 50, profile: str = "mixed"):
    print(f"\n{Colors.BOLD}>>> Launching Batch Stress Benchmark ({count} transactions, profile={profile})...{Colors.RESET}\n")

    t_start = time.time()
    latencies = []
    blocks = 0
    reviews = 0
    approves = 0
    redis_hits = 0
    sla_breaches = 0

    scenario_keys = list(SCENARIOS.keys())

    for idx in range(count):
        # Pick scenario based on profile
        if profile == "attack":
            key = "mule" if (idx % 3 != 0) else "retail"
        elif profile == "legitimate":
            key = "retail" if (idx % 4 != 0) else "payroll"
        else:
            key = scenario_keys[idx % len(scenario_keys)]

        payload = dict(SCENARIOS[key]["payload"])
        payload["transaction_id"] = f"TXN-BENCH-{idx+1:04d}"
        payload["timestamp"] = datetime.utcnow().isoformat() + "Z"

        res = score_transaction(endpoint, payload)
        if not res:
            continue

        lat = res.get("inference_latency_ms", 0.0)
        latencies.append(lat)
        dec = res.get("decision_preliminary", "APPROVE")
        if dec == "BLOCK":
            blocks += 1
        elif dec == "REVIEW":
            reviews += 1
        else:
            approves += 1

        if res.get("feature_source") == "redis":
            redis_hits += 1

        if lat > 200.0:
            sla_breaches += 1

        if (idx + 1) % 10 == 0 or (idx + 1) == count:
            sys.stdout.write(f"\r  Progress: {idx+1}/{count} transactions scored...")
            sys.stdout.flush()

    total_time = max(0.001, time.time() - t_start)
    latencies.sort()
    n = len(latencies)
    p50 = latencies[int(n * 0.50)] if n else 0
    p95 = latencies[int(n * 0.95)] if n else 0
    p99 = latencies[int(n * 0.99)] if n else 0
    avg_lat = sum(latencies) / max(1, n)
    tps = n / total_time
    sla_pct = ((n - sla_breaches) / max(1, n)) * 100

    print(f"\n\n{Colors.BOLD}=================== BENCHMARK SUMMARY ==================={Colors.RESET}")
    print(f"  Total Transactions Scored : {Colors.BOLD}{n}{Colors.RESET}")
    print(f"  Total Test Duration       : {total_time:.2f} seconds")
    print(f"  Throughput                : {Colors.CYAN}{Colors.BOLD}{tps:.1f} TPS{Colors.RESET} (Transactions/sec)")
    print(f"  Average Latency           : {avg_lat:.2f} ms")
    print(f"  p50 Latency (Median)      : {p50:.2f} ms")
    print(f"  p95 Latency               : {p95:.2f} ms")
    print(f"  p99 Latency               : {p99:.2f} ms")
    print(f"  SLA Compliance (<200ms)   : {Colors.GREEN if sla_pct >= 95 else Colors.YELLOW}{sla_pct:.1f}% ({sla_breaches} breaches){Colors.RESET}")
    print(f"  Redis Feature Hit Rate    : {(redis_hits / max(1, n)) * 100:.1f}%")
    print(f"  Decisions Breakdown       : {Colors.RED}{blocks} BLOCKED{Colors.RESET} | {Colors.YELLOW}{reviews} REVIEW{Colors.RESET} | {Colors.GREEN}{approves} APPROVED{Colors.RESET}")
    print("=========================================================\n")


def interactive_mode(endpoint: str):
    print(f"\n{Colors.BOLD}Entering Interactive AI Engine Console...{Colors.RESET}")
    print("Enter transaction parameters below (press Enter to use defaults):\n")

    tid = input("Transaction ID [TXN-MANUAL-001]: ").strip() or "TXN-MANUAL-001"
    from_acc = input("Originating Account [842E23DC0]: ").strip() or "842E23DC0"
    to_acc = input("Beneficiary Account [ACC-BENEFICIARY]: ").strip() or "ACC-BENEFICIARY"
    amt_str = input("Amount in USD [9500.00]: ").strip() or "9500.00"
    fmt = input("Payment Format (Wire, ACH, Bitcoin, Cash, etc.) [Wire]: ").strip() or "Wire"

    try:
        amt = float(amt_str)
    except ValueError:
        amt = 9500.00

    payload = {
        "transaction_id": tid,
        "from_account": from_acc,
        "to_account": to_acc,
        "amount": amt,
        "payment_format": fmt,
        "currency": "USD",
        "timestamp": datetime.utcnow().isoformat() + "Z",
    }

    res = score_transaction(endpoint, payload)
    if not res:
        return

    lgbm = res.get("lgbm_score", 0.0)
    ae = res.get("ae_score", 0.0)
    dec = res.get("decision_preliminary", "APPROVE")
    comp = 0.7 * lgbm + 0.3 * ae

    print(f"\n>>> Decision: {format_decision(dec)} | Composite Risk: {comp*100:.2f}% (LGBM: {lgbm*100:.2f}%, AE: {ae*100:.2f}%)")
    print(f"    Source: {res.get('feature_source')} | Latency: {res.get('inference_latency_ms')}ms\n")


def main():
    parser = argparse.ArgumentParser(description="MuleGuard.AI Enterprise AI Engine Simulator & Benchmark")
    parser.add_argument("--endpoint", default=DEFAULT_ENDPOINT, help=f"AI Engine URL (default: {DEFAULT_ENDPOINT})")
    parser.add_argument("--health", action="store_true", help="Ping /health probe and display loaded models status")
    parser.add_argument("--scenario", choices=list(SCENARIOS.keys()) + ["all"], help="Execute specific scenario or 'all'")
    parser.add_argument("--stress", action="store_true", help="Run latency & throughput stress benchmark")
    parser.add_argument("--count", type=int, default=30, help="Number of transactions for stress test (default: 30)")
    parser.add_argument("--profile", choices=["mixed", "attack", "legitimate"], default="mixed", help="Traffic profile for stress benchmark")
    parser.add_argument("--interactive", action="store_true", help="Launch interactive manual scoring prompt")
    parser.add_argument("--export", type=str, help="Export result to JSON file")

    args = parser.parse_args()
    print_banner()

    if args.health:
        display_health(args.endpoint)
    elif args.scenario == "all":
        run_all_scenarios(args.endpoint)
    elif args.scenario:
        run_scenario(args.endpoint, args.scenario, args.export)
    elif args.stress:
        run_stress_benchmark(args.endpoint, args.count, args.profile)
    elif args.interactive:
        interactive_mode(args.endpoint)
    else:
        # Default behavior: show health and run the primary mule burst scenario
        display_health(args.endpoint)
        run_scenario(args.endpoint, "mule")
        print("\nTip: Run with --help to see all benchmark and stress testing options.\n")


if __name__ == "__main__":
    main()
