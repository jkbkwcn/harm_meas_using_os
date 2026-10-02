#!/usr/bin/env python3
"""
ADC Frame Receiver with Enhanced Frame Tracking and Plotting
Receives frames, saves complete frame info (header + payload), and plots with frame boundaries.
"""

import serial
import struct
import time
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Optional


@dataclass
class FrameHeader:
    """Frame header structure matching STM32 frame_hdr_t"""

    sync0: int
    sync1: int
    frame_id: int
    length: int
    overflow_count: int


class FrameReceiver:
    """Handles frame reception and loss detection"""

    SYNC_BYTE_0 = 0xA5
    SYNC_BYTE_1 = 0x5A
    HEADER_SIZE = 8  # sync0 + sync1 + frame_id + len + overflow_count

    def __init__(self, port: str, baudrate: int = 115200, timeout: float = 1.0):
        """
        Initialize frame receiver

        Args:
            port: Serial port (e.g., 'COM3' on Windows, '/dev/ttyACM0' on Linux)
            baudrate: Baud rate (default 115200)
            timeout: Read timeout in seconds
        """
        self.ser = serial.Serial(port, baudrate, timeout=timeout)
        self.expected_frame_id: Optional[int] = None
        self.last_overflow_count: Optional[int] = None

        # Statistics
        self.total_frames_received = 0
        self.total_frames_expected = 0
        self.total_dropped = 0
        self.sync_errors = 0
        self.start_time = time.time()

    def find_sync(self):
        """
        Search for sync bytes in stream

        Returns:
            True if sync found, False on timeout
        """
        while True:
            b = self.ser.read(1)
            if len(b) == 0:
                return False  # Timeout

            if b[0] == self.SYNC_BYTE_0:
                b2 = self.ser.read(1)
                if len(b2) == 0:
                    return False

                if b2[0] == self.SYNC_BYTE_1:
                    return True

        return False

    def receive_frame(self):
        """
        Receive one complete frame

        Returns:
            Tuple of (FrameHeader, ADC samples) or None on error
        """
        # Find sync bytes
        if not self.find_sync():
            return None

        # Read header (without sync bytes)
        header_data = self.ser.read(6)  # frame_id + len + overflow_count
        if len(header_data) != 6:
            self.sync_errors += 1
            return None

        frame_id, data_len, overflow_count = struct.unpack("<HHH", header_data)

        # Validate data length
        if data_len > 4096:  # Sanity check
            self.sync_errors += 1
            return None

        # Read ADC data
        adc_data = self.ser.read(data_len)
        if len(adc_data) != data_len:
            self.sync_errors += 1
            return None

        # Create header object
        header = FrameHeader(
            sync0=self.SYNC_BYTE_0,
            sync1=self.SYNC_BYTE_1,
            frame_id=frame_id,
            length=data_len,
            overflow_count=overflow_count,
        )

        # Convert to ADC samples
        num_samples = data_len // 2
        samples = np.frombuffer(adc_data, dtype=np.uint16, count=num_samples)

        # Check for frame loss
        self._check_frame_loss(header)

        # Update statistics
        self.total_frames_received += 1
        self.total_frames_expected += 1

        return header, samples

    def _check_frame_loss(self, header: FrameHeader):
        """Check for dropped frames using frame_id and overflow_count"""

        # Check frame_id sequence
        if self.expected_frame_id is not None:
            expected = self.expected_frame_id

            if header.frame_id != expected:
                # Calculate gap (handle uint16 wraparound)
                if header.frame_id > expected:
                    gap = header.frame_id - expected
                else:
                    gap = (0xFFFF - expected) + header.frame_id + 1

                self.total_dropped += gap
                self.total_frames_expected += gap

                elapsed = time.time() - self.start_time
                print(f"\n⚠️  [{elapsed:.1f}s] FRAME LOSS DETECTED!")
                print(f"    Expected frame_id: {expected}")
                print(f"    Received frame_id: {header.frame_id}")
                print(f"    Gap: {gap} frame(s)")

        # Check MCU overflow counter
        if self.last_overflow_count is not None:
            if header.overflow_count > self.last_overflow_count:
                new_overflows = header.overflow_count - self.last_overflow_count
                elapsed = time.time() - self.start_time
                print(f"\n⚠️  [{elapsed:.1f}s] MCU OVERFLOW!")
                print(f"    MCU dropped {new_overflows} frame(s)")
                print(f"    Total MCU overflows: {header.overflow_count}")

        # Update expected values
        self.expected_frame_id = (header.frame_id + 1) & 0xFFFF
        self.last_overflow_count = header.overflow_count

    def get_stats(self):
        """
        Get reception statistics

        Returns:
            Dictionary with statistics
        """
        elapsed = time.time() - self.start_time

        if self.total_frames_expected == 0:
            loss_rate = 0.0
        else:
            loss_rate = (self.total_dropped / self.total_frames_expected) * 100

        if elapsed > 0:
            frame_rate = self.total_frames_received / elapsed
        else:
            frame_rate = 0.0

        return {
            "elapsed_time": elapsed,
            "frames_received": self.total_frames_received,
            "frames_dropped": self.total_dropped,
            "loss_rate": loss_rate,
            "frame_rate": frame_rate,
            "sync_errors": self.sync_errors,
        }

    def print_stats(self):
        """Print formatted statistics"""
        stats = self.get_stats()

        print(f"\n{'='*60}")
        print(f"Reception Statistics")
        print(f"{'='*60}")
        print(f"Elapsed time:      {stats['elapsed_time']:.1f} s")
        print(f"Frames received:   {stats['frames_received']}")
        print(f"Frames dropped:    {stats['frames_dropped']}")
        print(f"Loss rate:         {stats['loss_rate']:.3f}%")
        print(f"Frame rate:        {stats['frame_rate']:.1f} frames/s")
        print(f"Sync errors:       {stats['sync_errors']}")
        print(f"{'='*60}\n")

    def close(self):
        """Close serial port"""
        self.ser.close()


def plot_data(
    all_samples: np.ndarray,
    num_frames: int,
    stats: dict,
    frame_lengths: list = None,
    frame_ids: list = None,
    overflow_counts: list = None,
    skip_first_frames: int = 8,
):
    """
    Plot the received ADC data with frame boundaries

    Args:
        all_samples: Concatenated ADC samples from all frames
        num_frames: Number of frames received
        stats: Reception statistics dictionary
        frame_lengths: List of frame lengths (number of samples per frame)
        frame_ids: List of frame IDs
        overflow_counts: List of overflow counts per frame
        skip_first_frames: Number of initial frames that were skipped
    """
    # Create figure with multiple subplots
    fig, axes = plt.subplots(3, 1, figsize=(14, 11))

    # Time axis (sample indices)
    sample_indices = np.arange(len(all_samples))

    # Calculate frame boundaries and detect issues
    frame_boundaries = []
    frame_colors = []
    frame_labels = []

    if frame_lengths is not None and len(frame_lengths) > skip_first_frames:
        # Calculate cumulative positions (skip first frames)
        cumulative_pos = 0

        for i in range(skip_first_frames, len(frame_lengths)):
            # Check for frame loss (gap in frame IDs)
            frame_loss = False
            overflow_increase = False

            if frame_ids is not None and i > skip_first_frames:
                expected_id = (frame_ids[i - 1] + 1) & 0xFFFF
                if frame_ids[i] != expected_id:
                    frame_loss = True

            # Check for overflow increase
            if overflow_counts is not None and i > skip_first_frames:
                if overflow_counts[i] > overflow_counts[i - 1]:
                    overflow_increase = True

            # Determine color (red for problems, gray for normal)
            color = "red" if (frame_loss or overflow_increase) else "gray"

            frame_boundaries.append(cumulative_pos)
            frame_colors.append(color)
            frame_labels.append(frame_ids[i] if frame_ids is not None else i)
            cumulative_pos += frame_lengths[i]

    # Plot 1: All samples with frame boundaries
    axes[0].plot(sample_indices, all_samples, linewidth=0.5, zorder=1)

    # Add vertical lines for frame boundaries
    for pos, color, label in zip(frame_boundaries, frame_colors, frame_labels):
        axes[0].axvline(
            x=pos,
            color=color,
            alpha=0.6 if color == "red" else 0.3,
            linewidth=1.5 if color == "red" else 0.8,
            linestyle="-" if color == "red" else "--",
            zorder=2,
        )

    # Add frame ID labels (every N frames to avoid clutter)
    label_every = max(1, len(frame_boundaries) // 20)  # Show ~20 labels max
    for idx, (pos, color, label) in enumerate(zip(frame_boundaries, frame_colors, frame_labels)):
        if idx % label_every == 0 or color == "red":
            axes[0].text(
                pos,
                axes[0].get_ylim()[1] * 0.98,
                str(label),
                rotation=90,
                fontsize=7,
                color=color,
                alpha=0.8 if color == "red" else 0.5,
                verticalalignment="top",
                horizontalalignment="right",
            )

    axes[0].set_xlabel("Sample Index")
    axes[0].set_ylabel("ADC Value")
    axes[0].set_title(f"ADC Data - All {len(all_samples)} Samples ({num_frames} frames)\n" f"Red lines: Frame loss or overflow | Gray lines: Frame boundaries")
    axes[0].grid(True, alpha=0.3, zorder=0)

    # Plot 2: First 1000 samples (or all if less) with detailed frame markers
    num_to_plot = min(1000, len(all_samples))
    axes[1].plot(
        sample_indices[:num_to_plot],
        all_samples[:num_to_plot],
        marker="o",
        markersize=2,
        linewidth=0.8,
        zorder=1,
    )

    # Add vertical lines for frame boundaries (only those visible in this range)
    for pos, color, label in zip(frame_boundaries, frame_colors, frame_labels):
        if pos < num_to_plot:
            axes[1].axvline(
                x=pos,
                color=color,
                alpha=0.6 if color == "red" else 0.3,
                linewidth=1.5 if color == "red" else 0.8,
                linestyle="-" if color == "red" else "--",
                zorder=2,
            )
            # Label all frames in detailed view
            axes[1].text(
                pos,
                axes[1].get_ylim()[1] * 0.98,
                str(label),
                rotation=90,
                fontsize=8,
                color=color,
                alpha=0.9 if color == "red" else 0.6,
                verticalalignment="top",
                horizontalalignment="right",
            )

    axes[1].set_xlabel("Sample Index")
    axes[1].set_ylabel("ADC Value")
    axes[1].set_title(f"ADC Data - First {num_to_plot} Samples (Detailed View with Frame IDs)")
    axes[1].grid(True, alpha=0.3, zorder=0)

    # Plot 3: Histogram
    axes[2].hist(all_samples, bins=100, edgecolor="black", alpha=0.7)
    axes[2].set_xlabel("ADC Value")
    axes[2].set_ylabel("Count")
    axes[2].set_title("ADC Value Distribution")
    axes[2].grid(True, alpha=0.3, axis="y")

    # Add statistics text to the figure
    red_frames = sum(1 for c in frame_colors if c == "red")
    stats_text = (
        f"Frames received: {stats['frames_received']}\n" f"Frames dropped: {stats['frames_dropped']}\n" f"Loss rate: {stats['loss_rate']:.3f}%\n" f"Frame rate: {stats['frame_rate']:.1f} fps\n" f"Problematic frames: {red_frames}\n" f"Total samples: {len(all_samples)}\n" f"Min/Max: {all_samples.min()}/{all_samples.max()}\n" f"Mean: {all_samples.mean():.1f}\n" f"Std Dev: {all_samples.std():.1f}"
    )

    fig.text(
        0.02,
        0.98,
        stats_text,
        transform=fig.transFigure,
        fontsize=9,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )

    fig.tight_layout()
    plt.show()


def receive_and_plot(port: str, num_frames: int, baudrate: int = 115200, save_file: Optional[str] = None):
    """
    Receive specified number of frames and plot the data

    Args:
        port: Serial port (e.g., 'COM7')
        num_frames: Number of frames to receive before disconnecting
        baudrate: Baud rate (default 115200)
        save_file: Optional base filename to save data (e.g., 'adc_data')
    """
    print(f"Opening {port} at {baudrate} baud...")
    print(f"Target: {num_frames} frames\n")

    try:
        receiver = FrameReceiver(port, baudrate)
        print(f"✓ Connected!")
        print(f"Receiving frames...\n")

        frames_data = []
        frame_ids = []
        overflow_counts = []
        frame_lengths = []

        while receiver.total_frames_received < num_frames:
            result = receiver.receive_frame()

            if result is None:
                continue

            header, samples = result
            frames_data.append(samples)
            frame_ids.append(header.frame_id)
            overflow_counts.append(header.overflow_count)
            frame_lengths.append(len(samples))

            # Print progress
            progress = (receiver.total_frames_received / num_frames) * 100
            print(f"Frame {receiver.total_frames_received}/{num_frames} " f"({progress:.1f}%) | " f"ID: {header.frame_id:5d} | " f"Samples: {len(samples):3d} | " f"MCU overflow: {header.overflow_count:3d}")

        print(f"\n✓ Received {num_frames} frames!")

        # Print final statistics
        receiver.print_stats()

        # Concatenate all samples
        if len(frames_data) > 0:
            all_samples = np.concatenate(frames_data)
            all_samples_trimmed = all_samples[10 * 512 :]  # Cut first 10 frames (10*512 samples)
            print(f"Total samples collected: {len(all_samples)}")
            print(f"Samples after trimming: {len(all_samples_trimmed)}")

            # Save data if requested
            if save_file is not None:
                # Determine base filename (without extension)
                if save_file.endswith(".npy"):
                    base_file = save_file[:-4]
                elif save_file.endswith(".npz"):
                    base_file = save_file[:-4]
                else:
                    base_file = save_file

                # Save concatenated samples (for backward compatibility)
                samples_file = base_file + ".npy"
                print(f"\nSaving concatenated samples to {samples_file}...")
                np.save(samples_file, all_samples_trimmed)
                print(f"✓ Saved {len(all_samples_trimmed)} samples")

                # Save frames with complete metadata (header + payload)
                frames_file = base_file + "_frames.npz"
                print(f"Saving frames with metadata to {frames_file}...")

                # Convert frame metadata to arrays
                frame_ids_arr = np.array(frame_ids, dtype=np.uint16)
                overflow_counts_arr = np.array(overflow_counts, dtype=np.uint16)
                frame_lengths_arr = np.array(frame_lengths, dtype=np.uint16)

                # Create a dict to save individual frame payloads
                frame_payloads = {f"frame_{i}": frames_data[i] for i in range(len(frames_data))}

                # Save all frame data with metadata
                np.savez_compressed(
                    frames_file,
                    frame_ids=frame_ids_arr,
                    overflow_counts=overflow_counts_arr,
                    frame_lengths=frame_lengths_arr,
                    num_frames=len(frames_data),
                    **frame_payloads,
                )

                print(f"✓ Saved {len(frames_data)} frames with complete metadata")
                print(f"  - Frame IDs: {len(frame_ids)} entries")
                print(f"  - Overflow counts: {len(overflow_counts)} entries")
                print(f"  - Individual frame payloads preserved")
                print(f"\nTo restore frames, use: load_frames('{frames_file}') from load_frames.py")

            # Get stats for plotting
            stats = receiver.get_stats()

            # Close connection before plotting
            receiver.close()
            print("\nSerial port closed")

            # Plot the data with frame boundaries
            print("\nGenerating plots with frame boundaries...")
            plot_data(
                all_samples_trimmed,
                num_frames,
                stats,
                frame_lengths=frame_lengths,
                frame_ids=frame_ids,
                overflow_counts=overflow_counts,
                skip_first_frames=8,
            )
        else:
            print("No data to plot!")
            receiver.close()

    except KeyboardInterrupt:
        print("\n\nStopped by user")
        receiver.print_stats()
        receiver.close()
        print("\nSerial port closed")

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback

        traceback.print_exc()
        receiver.close()


def main():
    """Main function with configuration"""
    # ========== CONFIGURATION ==========
    PORT = "COM3"  # Change to your COM port
    NUM_FRAMES = 5000  # Number of frames to receive
    BAUDRATE = 115200  # Baud rate
    SAVE_FILE = "measured_data/adc_data_tri_sin_200uVpkpk_50Hz"  # Base filename (extensions added automatically)
    # ===================================

    receive_and_plot(PORT, NUM_FRAMES, BAUDRATE, SAVE_FILE)


if __name__ == "__main__":
    main()
