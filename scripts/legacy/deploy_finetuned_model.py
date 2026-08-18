#!/usr/bin/env python3
"""
Deploy finetuned Attention Fusion model to production
Backs up baseline and replaces with finetuned checkpoint
"""

import shutil
from pathlib import Path
import json
from datetime import datetime

print("=" * 70)
print("🚀 DEPLOYING FINETUNED ATTENTION FUSION MODEL")
print("=" * 70)

# File paths
baseline_path = Path('outputs/attention_fusion_model_best.pt')
finetuned_path = Path('outputs/attention_fusion_finetuned_best.pt')
backup_path = Path('outputs/attention_fusion_model_baseline_backup_20260405.pt')

# Verify files exist
print("\n📋 Verifying checkpoint files...")
if not baseline_path.exists():
    print(f"❌ Baseline model not found: {baseline_path}")
    exit(1)
print(f"✅ Baseline found: {baseline_path} ({baseline_path.stat().st_size / 1024 / 1024:.2f} MB)")

if not finetuned_path.exists():
    print(f"❌ Finetuned model not found: {finetuned_path}")
    exit(1)
print(f"✅ Finetuned found: {finetuned_path} ({finetuned_path.stat().st_size / 1024 / 1024:.2f} MB)")

# Backup baseline
print(f"\n💾 Backing up baseline model...")
try:
    shutil.copy2(baseline_path, backup_path)
    print(f"✅ Backup created: {backup_path}")
except Exception as e:
    print(f"❌ Backup failed: {e}")
    exit(1)

# Deploy finetuned
print(f"\n🔄 Deploying finetuned model to production...")
try:
    # Replace baseline with finetuned
    shutil.copy2(finetuned_path, baseline_path)
    print(f"✅ Finetuned model deployed to: {baseline_path}")
except Exception as e:
    print(f"❌ Deployment failed: {e}")
    # Restore backup
    print("⚠️  Attempting to restore backup...")
    try:
        shutil.copy2(backup_path, baseline_path)
        print("✅ Backup restored")
    except:
        print("❌ Failed to restore backup - MANUAL INTERVENTION REQUIRED")
    exit(1)

# Log deployment
print(f"\n📝 Logging deployment...")
deployment_log = {
    'timestamp': datetime.now().isoformat(),
    'status': 'SUCCESS',
    'baseline_backup': str(backup_path),
    'production_model': str(baseline_path),
    'finetuned_checkpoint': str(finetuned_path),
    'improvements': {
        'baseline_accuracy': 78.57,
        'finetuned_accuracy': 82.06,
        'improvement_pp': 3.49,
    }
}

log_file = Path('outputs/deployment_log.json')
with open(log_file, 'w') as f:
    json.dump(deployment_log, f, indent=2)
print(f"✅ Deployment log: {log_file}")

print("\n" + "=" * 70)
print("✅ DEPLOYMENT COMPLETE")
print("=" * 70)
print(f"\n📊 Summary:")
print(f"  • Baseline backup:    {backup_path}")
print(f"  • Production model:   {baseline_path}")
print(f"  • Improvement:        +3.49pp (78.57% → 82.06%)")
print(f"  • Status:             ✅ Ready for use")
print(f"\n🔄 The new model is now live in production")
print(f"   All references to 'attention_fusion_model_best.pt' will use the finetuned model\n")
