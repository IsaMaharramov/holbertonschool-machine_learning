#!/usr/bin/env python3
"""
Module to plot multiple subplots in a single figure
"""
import numpy as np
import matplotlib.pyplot as plt


def all_in_one():
    """
    Constructs a 3x2 grid integrating all previous graph exercises
    """
    y0 = np.arange(0, 11) ** 3

    mean = [69, 0]
    cov = [[15, 8], [8, 15]]
    np.random.seed(5)
    x1, y1 = np.random.multivariate_normal(mean, cov, 2000).T
    y1 += 180

    x2 = np.arange(0, 28651, 5730)
    r2 = np.log(0.5)
    t2 = 5730
    y2 = np.exp((r2 / t2) * x2)

    x3 = np.arange(0, 21000, 1000)
    r3 = np.log(0.5)
    t31 = 5730
    t32 = 1600
    y31 = np.exp((r3 / t31) * x3)
    y32 = np.exp((r3 / t32) * x3)

    np.random.seed(5)
    student_grades = np.random.normal(68, 15, 50)

    fig = plt.figure(figsize=(6.4, 4.8))
    fig.suptitle('All in One')

    # Plot 1: 0-line
    ax1 = fig.add_subplot(3, 2, 1)
    ax1.plot(y0, 'r-')
    ax1.set_xlim(0, 10)

    # Plot 2: 1-scatter
    ax2 = fig.add_subplot(3, 2, 2)
    ax2.scatter(x1, y1, color='magenta')
    ax2.set_xlabel('Height (in)', fontsize='x-small')
    ax2.set_ylabel('Weight (lbs)', fontsize='x-small')
    ax2.set_title("Men's Height vs Weight", fontsize='x-small')

    # Plot 3: 2-change_scale
    ax3 = fig.add_subplot(3, 2, 3)
    ax3.plot(x2, y2)
    ax3.set_xlabel('Time (years)', fontsize='x-small')
    ax3.set_ylabel('Fraction Remaining', fontsize='x-small')
    ax3.set_title('Exponential Decay of C-14', fontsize='x-small')
    ax3.set_xlim(0, 28650)
    ax3.set_yscale('log')

    # Plot 4: 3-two
    ax4 = fig.add_subplot(3, 2, 4)
    ax4.plot(x3, y31, 'r--', label='C-14')
    ax4.plot(x3, y32, 'g-', label='Ra-226')
    ax4.set_xlabel('Time (years)', fontsize='x-small')
    ax4.set_ylabel('Fraction Remaining', fontsize='x-small')
    ax4.set_title('Exponential Decay of Radioactive Elements',
                  fontsize='x-small')
    ax4.set_xlim(0, 20000)
    ax4.set_ylim(0, 1)
    ax4.legend(fontsize='x-small', loc='upper right')

    # Plot 5: 4-frequency
    ax5 = fig.add_subplot(3, 2, (5, 6))
    bins = np.arange(0, 101, 10)
    ax5.hist(student_grades, bins=bins, edgecolor='black')
    ax5.set_xlabel('Grades', fontsize='x-small')
    ax5.set_ylabel('Number of Students', fontsize='x-small')
    ax5.set_title('Project A', fontsize='x-small')
    ax5.set_xlim(0, 100)
    ax5.set_ylim(0, 30)
    ax5.set_xticks(np.arange(0, 101, 10))

    plt.tight_layout()
    plt.show()
