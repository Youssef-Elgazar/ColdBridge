import sys, traceback
sys.path.insert(0, '.')
try:
    from run_fixed_experiments import load_industry40, evaluate_module_a, compute_metrics, baseline_lstm, baseline_static_ttl, baseline_iat_threshold
    records = load_industry40()
    print(f"Loaded {len(records)} records")
    train = records[:378]
    test = records[378:]
    n_cold_train = sum(1 for r in train if r["was_cold"])
    n_cold_test = sum(1 for r in test if r["was_cold"])
    print(f"Train: {len(train)} ({n_cold_train} cold), Test: {len(test)} ({n_cold_test} cold)")
    
    print("\n--- Static TTL ---")
    p1 = baseline_static_ttl(test, ttl=600.0)
    print(compute_metrics(p1, "Static TTL"))
    
    print("\n--- IAT Threshold ---")
    p2 = baseline_iat_threshold(test, threshold_s=300.0)
    print(compute_metrics(p2, "IAT Threshold"))
    
    print("\n--- LSTM ---")
    p3 = baseline_lstm(train, test, epochs=10)
    if p3:
        print(compute_metrics(p3, "LSTM"))
    else:
        print("LSTM: insufficient data")
    
    print("\n--- ColdBridge Module A ---")
    preds, info = evaluate_module_a(train, test, epochs=10, theta=0.50)
    print(compute_metrics(preds, "ColdBridge"))
    
except Exception as e:
    traceback.print_exc()
