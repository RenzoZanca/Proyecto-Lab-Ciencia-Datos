#!/bin/bash

echo "🚀 Starting SodAI Complete System..."
echo "======================================"

# Step 1: Start Airflow
echo "📊 Step 1: Starting Airflow..."
cd airflow
docker compose up -d
echo "✅ Airflow starting..."

# Step 2: Start App (Backend + Frontend)
echo "🔧 Step 2: Starting Backend & Frontend..."
cd ../app
docker compose up -d
echo "✅ App services starting..."

# Step 3: Wait for services to be ready
echo "⏳ Step 3: Waiting for services to initialize..."
sleep 10

# Step 4: Check service health
echo "🔍 Step 4: Checking service health..."

# Check Airflow
echo -n "   Airflow (http://localhost:8080): "
if curl -s "http://localhost:8080/health" > /dev/null 2>&1; then
    echo "✅ Ready"
else
    echo "⚠️  Starting (may take a few more seconds)"
fi

# Check Backend
echo -n "   Backend (http://localhost:8000): "
if curl -s "http://localhost:8000/" > /dev/null 2>&1; then
    echo "✅ Ready"
else
    echo "❌ Not responding"
fi

# Check Frontend
echo -n "   Frontend (http://localhost:7860): "
if curl -s "http://localhost:7860/" > /dev/null 2>&1; then
    echo "✅ Ready"
else
    echo "❌ Not responding"
fi

# Step 5: Check and activate DAG
echo "🔄 Step 5: Checking DAG status..."
sleep 5  # Give backend time to auto-activate

DAG_STATUS=$(curl -s "http://localhost:8000/dag/status" 2>/dev/null)
if echo "$DAG_STATUS" | grep -q '"airflow_available": true'; then
    if echo "$DAG_STATUS" | grep -q '"is_paused": false'; then
        echo "   ✅ DAG is active (auto-activated by backend)"
    else
        echo "   🔄 DAG was paused, activating..."
        ACTIVATION_RESULT=$(curl -s -X POST "http://localhost:8000/dag/activate" 2>/dev/null)
        if echo "$ACTIVATION_RESULT" | grep -q '"success": true'; then
            echo "   ✅ DAG activated successfully"
        else
            echo "   ⚠️  DAG activation failed - you may need to activate manually"
        fi
    fi
else
    echo "   ⚠️  Airflow not ready yet - DAG will auto-activate when ready"
fi

echo ""
echo "🎉 SodAI System Status:"
echo "======================"
echo "   Frontend:  http://localhost:7860"
echo "   Backend:   http://localhost:8000"
echo "   Airflow:   http://localhost:8080"
echo ""
echo "📋 Available API endpoints:"
echo "   GET  /dag/status     - Check DAG status"
echo "   POST /dag/activate   - Manually activate DAG"
echo "   POST /upload_and_start - Start processing"
echo ""
echo "✨ The system will auto-activate the DAG when ready!"
echo "   You can now upload parquet files at http://localhost:7860" 