"""
Log formatting utilities for better visual organization.
"""

import logging
import sys
from typing import Optional


class Colors:
    """ANSI color codes for terminal output."""
    # Standard colors
    RED = '\033[91m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    MAGENTA = '\033[95m'
    CYAN = '\033[96m'
    WHITE = '\033[97m'
    
    # Bright colors
    BRIGHT_RED = '\033[1;91m'
    BRIGHT_GREEN = '\033[1;92m'
    BRIGHT_YELLOW = '\033[1;93m'
    BRIGHT_BLUE = '\033[1;94m'
    BRIGHT_MAGENTA = '\033[1;95m'
    BRIGHT_CYAN = '\033[1;96m'
    
    # Styles
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    RESET = '\033[0m'
    
    # Background colors
    BG_RED = '\033[41m'
    BG_GREEN = '\033[42m'
    BG_YELLOW = '\033[43m'
    BG_BLUE = '\033[44m'


def create_section_header(title: str, color: str = Colors.BRIGHT_CYAN, width: int = 80) -> str:
    """Create a visually distinct section header."""
    border = "=" * width
    title_line = f" {title} "
    padding = (width - len(title_line)) // 2
    centered_title = "=" * padding + title_line + "=" * (width - padding - len(title_line))
    
    return f"\n{color}{border}\n{centered_title}\n{border}{Colors.RESET}\n"


def create_subsection_header(title: str, color: str = Colors.BRIGHT_YELLOW, width: int = 60) -> str:
    """Create a subsection header."""
    border = "-" * width
    title_line = f" {title} "
    padding = (width - len(title_line)) // 2
    centered_title = "-" * padding + title_line + "-" * (width - padding - len(title_line))
    
    return f"\n{color}{centered_title}{Colors.RESET}\n"


def create_step_header(step_num: int, title: str, color: str = Colors.BRIGHT_BLUE) -> str:
    """Create a numbered step header."""
    return f"\n{color}📋 Step {step_num}: {title}{Colors.RESET}"


def create_info_box(title: str, content: str, color: str = Colors.CYAN) -> str:
    """Create an information box that accommodates long paths without breaking them."""
    # Split content into lines but don't break them - let the box expand
    lines = content.split('\n')
    
    # Calculate width based on longest line
    max_content_len = max(len(line) for line in lines) if lines else 0
    title_len = len(title)
    max_len = max(max_content_len, title_len)
    
    # No maximum width limit - allow the box to expand to fit any content
    # This ensures full paths are always displayed without truncation
    width = max_len + 4  # +4 for padding and borders
    
    # Ensure minimum width for aesthetics
    width = max(width, 40)
    
    top_border = "┌" + "─" * (width - 2) + "┐"
    bottom_border = "└" + "─" * (width - 2) + "┘"
    title_line = f"│ {title:<{width-4}} │"
    separator = "├" + "─" * (width - 2) + "┤"
    
    content_lines = []
    for line in lines:
        # No truncation - display full line
        content_lines.append(f"│ {line:<{width-4}} │")
    
    box = f"\n{color}{top_border}\n{title_line}\n{separator}\n"
    box += "\n".join(content_lines)
    box += f"\n{bottom_border}{Colors.RESET}\n"
    
    return box


def log_section_start(title: str, color: str = Colors.BRIGHT_CYAN):
    """Log the start of a major section."""
    logging.info(create_section_header(title, color))


def log_subsection_start(title: str, color: str = Colors.BRIGHT_YELLOW):
    """Log the start of a subsection."""
    logging.info(create_subsection_header(title, color))


def log_step(step_num: int, title: str, color: str = Colors.BRIGHT_BLUE):
    """Log a numbered step."""
    logging.info(create_step_header(step_num, title, color))


def log_success(message: str):
    """Log a success message with green color."""
    logging.info(f"{Colors.BRIGHT_GREEN}✅ {message}{Colors.RESET}")


def log_warning(message: str):
    """Log a warning message with yellow color."""
    logging.warning(f"{Colors.BRIGHT_YELLOW}⚠️  {message}{Colors.RESET}")


def log_error(message: str):
    """Log an error message with red color."""
    logging.error(f"{Colors.BRIGHT_RED}❌ {message}{Colors.RESET}")


def log_info(message: str, color: str = Colors.WHITE):
    """Log an info message with optional color."""
    logging.info(f"{color}ℹ️  {message}{Colors.RESET}")


def log_progress(current: int, total: int, description: str):
    """Log progress with a progress indicator."""
    percentage = (current / total) * 100
    bar_length = 20
    filled_length = int(bar_length * current / total)
    bar = "█" * filled_length + "░" * (bar_length - filled_length)
    
    logging.info(f"{Colors.BRIGHT_CYAN}📊 {description}: [{bar}] {percentage:.1f}% ({current}/{total}){Colors.RESET}")


def log_metric(name: str, value: float, format_str: str = ".4f", color: str = Colors.BRIGHT_GREEN):
    """Log a metric with formatting."""
    logging.info(f"{color}📈 {name}: {value:{format_str}}{Colors.RESET}")


def log_config_item(key: str, value, color: str = Colors.CYAN):
    """Log a configuration item."""
    logging.info(f"{color}⚙️  {key}: {value}{Colors.RESET}")


def log_checkpoint(message: str):
    """Log a checkpoint message."""
    logging.info(f"\n{Colors.BRIGHT_MAGENTA}🔄 CHECKPOINT: {message}{Colors.RESET}\n")


def log_final_result(message: str):
    """Log final results with emphasis."""
    logging.info(f"\n{Colors.BG_GREEN}{Colors.BOLD} 🎯 FINAL RESULT: {message} {Colors.RESET}")


class ColoredFormatter(logging.Formatter):
    """Custom formatter that adds colors to log levels."""
    
    COLORS = {
        'DEBUG': Colors.CYAN,
        'INFO': Colors.WHITE,
        'WARNING': Colors.YELLOW,
        'ERROR': Colors.RED,
        'CRITICAL': Colors.BRIGHT_RED,
    }
    
    def format(self, record):
        # Store original format
        original_format = self._style._fmt
        
        # Apply color based on log level
        color = self.COLORS.get(record.levelname, Colors.WHITE)
        colored_levelname = f"{color}{record.levelname}{Colors.RESET}"
        
        # Update the format to include colored level name
        record.levelname = colored_levelname
        
        # Format the message
        formatted = super().format(record)
        
        return formatted


def setup_colored_logging():
    """Setup colored logging for better visibility."""
    # Get the root logger
    logger = logging.getLogger()
    
    # Check if we've already set up colored logging
    for handler in logger.handlers:
        if isinstance(handler, logging.StreamHandler) and hasattr(handler, 'formatter'):
            if isinstance(handler.formatter, ColoredFormatter):
                # Already set up, skip
                return
    
    # Only modify handlers that go to console and don't already have colored formatting
    for handler in logger.handlers:
        if isinstance(handler, logging.StreamHandler) and handler.stream in (sys.stdout, sys.stderr):
            # Create colored formatter
            formatter = ColoredFormatter(
                fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
            handler.setFormatter(formatter)


def log_epoch_train(epoch: int, loss: float, acc: float, balanced_acc: float, lr: float):
    """Log training epoch results in a formatted way."""
    logging.info(f"{Colors.BRIGHT_BLUE}🔹 Epoch {epoch:2d} [TRAIN]{Colors.RESET} "
                f"{Colors.GREEN}Loss: {loss:.4f}{Colors.RESET} | "
                f"{Colors.CYAN}Acc: {acc:.4f}{Colors.RESET} | "
                f"{Colors.MAGENTA}Bal.Acc: {balanced_acc:.4f}{Colors.RESET} | "
                f"{Colors.YELLOW}LR: {lr:.8f}{Colors.RESET}")


def log_epoch_val(epoch: int, loss: float, acc: float, balanced_acc: float, num_samples: int, is_best: bool = False, best_indicator: str = ""):
    """Log validation epoch results in a formatted way."""
    # Add best model indicator if this is the best epoch
    best_suffix = ""
    if is_best and best_indicator:
        best_suffix = f" {Colors.BRIGHT_GREEN}✓ {best_indicator}{Colors.RESET}"
    elif is_best:
        best_suffix = f" {Colors.BRIGHT_GREEN}✓ BEST{Colors.RESET}"
    
    logging.info(f"{Colors.BRIGHT_CYAN}🔸 Epoch {epoch:2d} [VAL  ]{Colors.RESET} "
                f"{Colors.GREEN}Loss: {loss:.4f}{Colors.RESET} | "
                f"{Colors.CYAN}Acc: {acc:.4f}{Colors.RESET} | "
                f"{Colors.MAGENTA}Bal.Acc: {balanced_acc:.4f}{Colors.RESET} | "
                f"{Colors.WHITE}Samples: {num_samples}{Colors.RESET}{best_suffix}")


def log_epoch_test(epoch: int, loss: float, acc: float, balanced_acc: float, num_samples: int, test_type: str = ""):
    """Log test epoch results in a formatted way with light red color."""
    # Format test label to match other labels (5 chars + brackets = 7 total)
    if test_type == "NEXT":
        test_label = "TEST+"  # TEST+ for next checkpoint
    elif test_type == " CUR":
        test_label = "TEST "  # TEST  for current checkpoint  
    else:
        test_label = "TEST "  # Default TEST
    
    logging.info(f"{Colors.RED}🔻 Epoch {epoch:2d} [{test_label}]{Colors.RESET} "
                f"{Colors.GREEN}Loss: {loss:.4f}{Colors.RESET} | "
                f"{Colors.CYAN}Acc: {acc:.4f}{Colors.RESET} | "
                f"{Colors.MAGENTA}Bal.Acc: {balanced_acc:.4f}{Colors.RESET} | "
                f"{Colors.WHITE}Samples: {num_samples}{Colors.RESET}")


def log_training_header(checkpoint: str, total_epochs: int):
    """Log a header for training session."""
    header = f"🚀 Training {checkpoint} - {total_epochs} epochs"
    logging.info(f"\n{Colors.BRIGHT_GREEN}{'='*60}\n{header:^60}\n{'='*60}{Colors.RESET}")


def log_training_summary(best_epoch: int, best_metric: float, metric_name: str = "balanced_acc"):
    """Log training completion summary."""
    logging.info(f"{Colors.BRIGHT_GREEN}✅ Training Complete{Colors.RESET} | "
                f"{Colors.BOLD}Best Epoch: {best_epoch}{Colors.RESET} | "
                f"{Colors.BOLD}Best {metric_name}: {best_metric:.4f}{Colors.RESET}")


def create_epoch_table_header():
    """Create a table header for epoch results that aligns with logging output."""
    # Use logging.info to match the format of the data rows
    header_line = f"{Colors.BOLD}{'Epoch':>5} | {'Phase':>5} | {'Loss':>8} | {'Acc':>8} | {'Bal.Acc':>8} | {'LR/Samples':>12}{Colors.RESET}"
    separator_line = f"{Colors.WHITE}{'-'*70}{Colors.RESET}"
    
    logging.info(header_line)
    logging.info(separator_line)
    return ""  # Return empty since we're logging directly


def log_epoch_table_row(epoch: int, phase: str, loss: float, acc: float, balanced_acc: float, extra_info: str):
    """Log a single epoch result as a table row."""
    phase_color = Colors.BRIGHT_BLUE if phase == "TRAIN" else Colors.BRIGHT_CYAN
    # Format exactly to match the header spacing
    row = f"{Colors.WHITE}{epoch:5d}{Colors.RESET} | " \
          f"{phase_color}{phase:>5}{Colors.RESET} | " \
          f"{Colors.GREEN}{loss:8.4f}{Colors.RESET} | " \
          f"{Colors.CYAN}{acc:8.4f}{Colors.RESET} | " \
          f"{Colors.MAGENTA}{balanced_acc:8.4f}{Colors.RESET} | " \
          f"{Colors.YELLOW}{extra_info:>12}{Colors.RESET}"
    
    logging.info(row)
