#!/usr/bin/env python3
"""
Simple frame monitor for quick testing
"""

import serial
import struct
import time

def main():
    # Configuration
    PORT = 'COM7'  # Change this to your port
    BAUDRATE = 115200
    
    print(f"Opening {PORT}...")
    ser = serial.Serial(PORT, BAUDRATE, timeout=1)
    print("Connected! Waiting for frames...\n")
    
    last_frame_id = None
    last_overflow = None
    frame_count = 0
    dropped_total = 0
    start_time = time.time()
    
    try:
        while True:
            # Find sync bytes
            while True:
                b = ser.read(1)
                if len(b) == 0:
                    continue
                if b[0] == 0xA5:
                    if ser.read(1)[0] == 0x5A:
                        break
            
            # Read header
            header = ser.read(6)
            if len(header) != 6:
                continue
            
            frame_id, length, overflow = struct.unpack('<HHH', header)
            
            # Read data
            data = ser.read(length)
            if len(data) != length:
                continue
            
            frame_count += 1
            
            # Check for dropped frames
            if last_frame_id is not None:
                expected = (last_frame_id + 1) & 0xFFFF
                if frame_id != expected:
                    gap = (frame_id - expected) & 0xFFFF
                    dropped_total += gap
                    print(f"⚠️  DROPPED {gap} frame(s)! (Expected {expected}, got {frame_id})")
            
            # Check MCU overflow
            if last_overflow is not None and overflow > last_overflow:
                new_drops = overflow - last_overflow
                print(f"⚠️  MCU OVERFLOW: {new_drops} frame(s)")
            
            # Update
            last_frame_id = frame_id
            last_overflow = overflow
            
            # Print status every 100 frames
            if frame_count % 100 == 0:
                elapsed = time.time() - start_time
                rate = frame_count / elapsed if elapsed > 0 else 0
                loss = (dropped_total / (frame_count + dropped_total)) * 100 if frame_count > 0 else 0
                
                print(f"Frame {frame_id:5d} | "
                      f"Samples: {length//2:3d} | "
                      f"MCU overflows: {overflow:3d} | "
                      f"Loss: {loss:.2f}% | "
                      f"Rate: {rate:.1f} fps")
    
    except KeyboardInterrupt:
        print("\n\nStopped")
        elapsed = time.time() - start_time
        print(f"\nReceived: {frame_count} frames")
        print(f"Dropped: {dropped_total} frames")
        print(f"Time: {elapsed:.1f} s")
        if frame_count > 0:
            print(f"Rate: {frame_count/elapsed:.1f} fps")
    
    finally:
        ser.close()

if __name__ == '__main__':
    main()
