# Sub-LSB Harmonics Measurement System

An electronic circuit project designed to measure mains current harmonics with amplitudes below one Least Significant Bit (LSB) of an Analog-to-Digital Converter.

This project was developed as part of the *Analog Peripheral Circuits in Digital Systems* course at the AGH University of Science and Technology in Kraków.

## Description

The main goal of this project was to design and test an analog-to-digital processing path capable of detecting very weak signals (e.g., 400 µVpp) using a converter with a relatively low base resolution (where 1 LSB ≈ 800 µV). To achieve this, digital signal processing techniques were implemented: oversampling and hardware-based triangular dithering.

The system measures harmonics up to the 5th order (250 Hz) for a 50 Hz mains signal, utilizing an STM32 microcontroller and a dedicated analog signal conditioning circuit on a custom PCB.

## Technologies and Specifications

* Microcontroller: STM32F103C8T6 (ARM Cortex-M3 core)
* ADC: Built-in 12-bit converter
* DSP Methods:
  * Oversampling enabling a theoretical improvement in the Effective Number of Bits (ENOB)
  * Triangular dithering generated via hardware PWM
* Hardware: Custom PCB designed in KiCad, featuring an analog front-end and USB power

## Key Findings and Measurement Results

The system successfully measures signals with amplitudes significantly below the 1 LSB threshold of the converter. Measurements of a 400 µVpp signal (at 50 Hz) yielded consistent results in FFT spectral analysis.

Key observation from the research:
Measurements revealed that the inherent white noise of the ADC and the analog path (amounting to about 2-3 LSB) was already sufficient to act as a natural broadband dither. This intrinsic noise successfully enabled the oversampling process, yielding an ENOB gain of approximately 4 bits. However, due to the ADC's intrinsic noise, the additionally injected triangular dither did not perform as theoretically expected. The built-in white noise essentially overshadowed the triangular signal, negating any extra theoretical benefits the triangle dither would normally provide over white noise. Despite this hardware limitation, the project successfully validated the oversampling concept and fully achieved its measurement objectives.