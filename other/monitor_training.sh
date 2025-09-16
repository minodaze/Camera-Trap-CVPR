#!/bin/bash

# Multi-GPU Training Status Monitor
# Shows current status of all GPU trainings

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
MAGENTA='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m'

clear
echo -e "${CYAN}🔍 Multi-GPU Training Status Monitor${NC}"
echo "====================================="
echo "$(date)"
echo ""

# Check if training processes are running
echo -e "${YELLOW}🔄 Running Training Processes:${NC}"
PROCESSES=$(pgrep -f "run_pipeline.py" | wc -l)
if [ $PROCESSES -eq 0 ]; then
    echo -e "${RED}  No training processes found${NC}"
else
    echo -e "${GREEN}  Found $PROCESSES training process(es)${NC}"
    
    # Show details of running processes
    echo ""
    echo -e "${BLUE}Process Details:${NC}"
    ps aux | grep "run_pipeline.py" | grep -v grep | while read line; do
        PID=$(echo $line | awk '{print $2}')
        CUDA_DEV=$(echo $line | grep -o "CUDA_VISIBLE_DEVICES=[0-9]" | cut -d= -f2)
        echo -e "${GREEN}  PID: $PID, GPU: $CUDA_DEV${NC}"
    done
fi

echo ""

# Check GPU usage
echo -e "${YELLOW}🖥️  GPU Status:${NC}"
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv,noheader,nounits | while IFS=, read gpu_id name util mem_used mem_total temp; do
        # Remove leading/trailing spaces
        gpu_id=$(echo $gpu_id | xargs)
        name=$(echo $name | xargs)
        util=$(echo $util | xargs)
        mem_used=$(echo $mem_used | xargs)
        mem_total=$(echo $mem_total | xargs)
        temp=$(echo $temp | xargs)
        
        # Color code based on utilization
        if [ $util -gt 80 ]; then
            color=$GREEN
        elif [ $util -gt 50 ]; then
            color=$YELLOW
        else
            color=$RED
        fi
        
        echo -e "  GPU $gpu_id: ${color}${util}%${NC} util, ${mem_used}/${mem_total}MB, ${temp}°C"
    done
else
    echo -e "${RED}  nvidia-smi not available${NC}"
fi

echo ""

# Check log files
echo -e "${YELLOW}📁 Recent Log Activity:${NC}"
if [ -f "multi_gpu_training.log" ]; then
    echo -e "${GREEN}  Main log (last 3 lines):${NC}"
    tail -n 3 multi_gpu_training.log | sed 's/^/    /'
else
    echo -e "${RED}  Main log file not found${NC}"
fi

echo ""

# Check GPU-specific logs
echo -e "${YELLOW}📊 GPU Training Progress:${NC}"
for gpu in 0 1 2 3; do
    LOG_DIR="logs/gpu_${gpu}_*"
    if ls $LOG_DIR 2>/dev/null >/dev/null; then
        DATASET=$(ls -d logs/gpu_${gpu}_* 2>/dev/null | head -1 | cut -d_ -f3-)
        LATEST_LOG=$(ls $LOG_DIR/training_*.out 2>/dev/null | tail -1)
        if [ -n "$LATEST_LOG" ]; then
            echo -e "${GREEN}  GPU $gpu ($DATASET):${NC} Active ($(basename $LATEST_LOG))"
        else
            echo -e "${YELLOW}  GPU $gpu ($DATASET):${NC} Log dir exists, no recent training"
        fi
    else
        echo -e "${RED}  GPU $gpu:${NC} No log directory found"
    fi
done

echo ""

# Training completion summary
echo -e "${YELLOW}📈 Training Summary:${NC}"
if [ -f "multi_gpu_training.log" ]; then
    COMPLETED=$(grep -c "completed successfully" multi_gpu_training.log 2>/dev/null || echo "0")
    FAILED=$(grep -c "failed with" multi_gpu_training.log 2>/dev/null || echo "0")
    TOTAL=$((COMPLETED + FAILED))
    
    echo -e "  ${GREEN}✅ Completed: $COMPLETED${NC}"
    echo -e "  ${RED}❌ Failed: $FAILED${NC}"
    echo -e "  📊 Total: $TOTAL/16 expected"
    
    if [ $TOTAL -eq 16 ]; then
        if [ $FAILED -eq 0 ]; then
            echo -e "  ${GREEN}🎉 All trainings completed successfully!${NC}"
        else
            echo -e "  ${YELLOW}⚠️  Some trainings failed${NC}"
        fi
    fi
else
    echo -e "${RED}  No training log found${NC}"
fi

echo ""
echo "====================================="
echo -e "${CYAN}Run 'watch -n 30 ./monitor_training.sh' for continuous monitoring${NC}"
