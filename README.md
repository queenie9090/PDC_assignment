Before running the project, please ensure that the following software has been installed:

CUDA: https://developer.nvidia.com/cuda-downloads \
MPI: https://www.microsoft.com/en-us/download/details.aspx?id=105289 \
OpenCV: https://sourceforge.net/projects/opencvlibrary/files/4.11.0/opencv-4.11.0-windows.exe/download

## Project Setup in Visual Studio:
1. Change Solution configuration to Release and Solution Configuration to x64
<img width="277" height="46" alt="image" src="https://github.com/user-attachments/assets/3a3d518d-d27f-473f-8f8c-f02fa2b1997f" />

## Configuration Setup for OpenCV
1. Right Click on Project in the top taskbar -> Properties
2. Go to VC++ Directories -> Include Directories -> Edit
3. Add path: C:\opencv\build\include (Ensure "Inherit from parent or project defaults" is checked) -> OK
<img width="648" height="464" alt="image" src="https://github.com/user-attachments/assets/03872895-0280-40bb-be22-46e833c5a435" />

4. Library Directories -> Edit.
5. Add path: C:\opencv\build\x64\vc16\lib (Ensure "Inherit from parent or project defaults" is checked) -> OK
<img width="696" height="479" alt="image" src="https://github.com/user-attachments/assets/4b32ee6e-3664-491c-9b61-87f8bb28cf37" />
 
6. Go to Linker -> Input -> Additional Dependencies -> Edit
7. Add opencv_world4110.lib for Release mode. Then, click Apply and OK
<img width="466" height="464" alt="image" src="https://github.com/user-attachments/assets/036dbc5c-f73a-4967-868a-c54db1d29c13" />

8. Add System Environment Variables: Press the Windows key, type env, and click Environment Variables...
<img width="541" height="565" alt="image" src="https://github.com/user-attachments/assets/3b32915f-7394-4643-94b8-8ad533ceb206" />

9. Edit the Path variable and add the following paths: C:\opencv\build\x64\vc16\lib and C:\opencv\build\x64\vc16\bin
<img width="842" height="387" alt="image" src="https://github.com/user-attachments/assets/20607c4d-e6e1-4baf-a293-7ba4098ff757" />
10. Move these two entries to the top of the list to give them highest priority.
<img width="722" height="671" alt="image" src="https://github.com/user-attachments/assets/9f6d5836-18af-4689-9bbc-406d8b70fff8" />

** Video Reference for openCV Setup: https://youtu.be/YUjamcyuKT4?si=0LInaGmP62oRuCGv **

## Configuration Setup for CUDA:  
1. Right Click on Project -> Build Dependecies -> Build Customizations. Mark check for Cuda 13.3 (or your current version) and Click OK
<img width="922" height="139" alt="image" src="https://github.com/user-attachments/assets/9b61f040-4960-428c-86a6-500f55e49cd0" />

2. Right Click on Project -> Properties -> Linker > Input > Additional Properties > edit
<img width="981" height="398" alt="image" src="https://github.com/user-attachments/assets/d61b610c-baaa-4eb0-8640-f81536fa533c" />

3. Add library cudart_static.lib and click OK and click Apply.
<img width="469" height="388" alt="image" src="https://github.com/user-attachments/assets/7d42b97c-14ae-4cfb-aa89-22ba74a04f07" />


## Configuration Setup for OpenMP: 
1. Right Click on Project in the top taskbar -> Properties 
2. Click C/C++ -> Language
3. Find OpenMP Support and change the dropdown value to Yes (/openmp) 
4. Click Apply and then OK 
<img width="792" height="304" alt="image" src="https://github.com/user-attachments/assets/c38ab395-d09d-4755-8027-78c68c6eccce" />


## Configuration Setup for MPI: 
1. Right Click on Project in the top taskbar -> Properties 
2. Navigate to C/C++ -> General 
3. Click on Additional Include Directories -> Edit. 
4. Add the following path: \Program Files (x86)\Microsoft SDKs\MPI\Include (Ensure Inherit from parent or project defaults is checked) -> OK
<img width="996" height="677" alt="WhatsApp Image 2026-08-10 at 12 25 16 AM" src="https://github.com/user-attachments/assets/16d3fdbc-9b50-4be7-8839-86ac190d70f4" />

5. Go to Linker -> General -> Additional Library Directories -> Edit 
6. Add the following path:\Program Files (x86)\Microsoft SDKs\MPI\Lib\x64 (Ensure Inherit from parent or project defaults is checked) -> OK
<img width="997" height="795" alt="WhatsApp Image 2026-08-10 at 12 25 43 AM" src="https://github.com/user-attachments/assets/1d265e47-13f4-416c-91a9-10e63fcc7b54" />

7. Go to Linker -> Input -> Additional Dependencies -> Edit 
8. Add msmpi.lib (Ensure Inherit from parent or project defaults is checked) -> OK 
9. Click Apply and OK
<img width="988" height="676" alt="WhatsApp Image 2026-08-10 at 12 26 05 AM" src="https://github.com/user-attachments/assets/7daa13a8-2018-4bf2-9b92-c23b08e742c7" />

<br><br><br>
## How to run the program
Before run the program, click Build -> Build Solution or Rebuild Solution
To run the program, Open Terminal using CTRL + ` or View -> Terminal
Command To run:
1. cd PDC_Assignment
2. python run_test_all.py       (Run all implementations)
3. python run_test_baseline.py  (Run Sequential Baseline only)
4. python run_test_omp.py       (Run OpenMP only)
5. python run_test_mpi.py       (Run MPI only)
6. python run_test_cuda.py      (Run CUDA only)

**Remark: If you plan to run only one parallel implementation (OpenMP, MPI, or CUDA), please run the baseline benchmark first. The baseline results are required to calculate the speedup; otherwise, the CSV file may contain missing or incomplete speedup values.

If you encounter a ModuleNotFoundError when running the Python scripts, install the required Python packages using:
1. pip install pandas
2. pip install streamlit
3. pip install opencv-python
4. pip install numpy

