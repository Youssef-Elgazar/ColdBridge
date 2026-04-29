from coldbridge.data.real_trace_loaders import Industry40ColdStartLoader
from coldbridge.modules.module_a import ModuleA

loader = Industry40ColdStartLoader()
records = loader.load_training_records()

train_history = []
for r in records:
    train_history.append({
        "function_name": r.function_name,
        "timestamp": r.timestamp,
        "was_cold": r.was_cold,
        "cold_start_latency_ms": r.cold_start_latency_ms,
    })

b_size = min(256, max(16, len(train_history) // 10))
print(f"Len: {len(train_history)}, b_size: {b_size}")

mod_a = ModuleA(theta=0.50)
mod_a.train(train_history, epochs=1, batch_size=b_size)
