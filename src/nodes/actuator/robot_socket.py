"""
Robot Socket Client - Send commands to robot server via TCP socket
"""
import logging
import os
import socket
from typing import Optional, Tuple

logger = logging.getLogger(__name__)

# Default configuration (can be overridden by environment variables)
DEFAULT_ROBOT_HOST = "192.168.0.111"
DEFAULT_ROBOT_PORT = 12345
SOCKET_TIMEOUT = 5.0  # seconds


def get_robot_config() -> Tuple[str, int]:
    """
    Get robot server configuration from environment variables or defaults.
    
    Returns:
        Tuple of (host, port)
    """
    host = os.getenv("ROBOT_SERVER_HOST", DEFAULT_ROBOT_HOST)
    port_str = os.getenv("ROBOT_SERVER_PORT", str(DEFAULT_ROBOT_PORT))
    try:
        port = int(port_str)
    except ValueError:
        logger.warning(f"Invalid ROBOT_SERVER_PORT value: {port_str}, using default: {DEFAULT_ROBOT_PORT}")
        port = DEFAULT_ROBOT_PORT
    return host, port


def send_robot_command(command: str, host: Optional[str] = None, port: Optional[int] = None) -> bool:
    """
    Send command to robot server via TCP socket.
    
    Args:
        command: Command string to send (e.g., "$SEQ;FWD,1;STOP\n")
        host: Robot server host (defaults to ROBOT_SERVER_HOST env var or DEFAULT_ROBOT_HOST)
        port: Robot server port (defaults to ROBOT_SERVER_PORT env var or DEFAULT_ROBOT_PORT)
    
    Returns:
        True if command was sent successfully, False otherwise
    """
    if host is None or port is None:
        host, port = get_robot_config()
    
    try:
        # Create socket connection
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(SOCKET_TIMEOUT)
        
        logger.info(f"Connecting to robot server at {host}:{port}...")
        sock.connect((host, port))
        
        # Send command
        logger.info(f"Sending command to robot: {repr(command)}")
        sock.sendall(command.encode('utf-8'))
        
        # Close connection
        sock.close()
        
        logger.info(f"Command sent successfully to {host}:{port}")
        return True
        
    except socket.timeout:
        logger.error(f"Timeout connecting to robot server at {host}:{port}")
        return False
    except socket.error as e:
        logger.error(f"Socket error connecting to robot server at {host}:{port}: {e}")
        return False
    except Exception as e:
        logger.exception(f"Unexpected error sending command to robot server: {e}")
        return False

