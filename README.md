# 3D Reconstruction with Spatial Memory

### [Paper](https://arxiv.org/abs/2408.16061) | [Project Page](https://hengyiwang.github.io/projects/spanner) | [Video](https://hengyiwang.github.io/projects/spanner/videos/spanner_intro.mp4)

> 3D Reconstruction with Spatial Memory <br />
> [Hengyi Wang](https://hengyiwang.github.io/), [Lourdes Agapito](http://www0.cs.ucl.ac.uk/staff/L.Agapito/)<br />
> arXiv 2024

<p align="center">
  <a href="">
    <img src="./assets/spann3r_teaser_white.gif" alt="Logo" width="90%">
  </a>
</p>

## Update

[2024-10-25] Add support for [Nerfstudio](assets/spanner-gs.gif)

[2024-10-18] Add camera param estimation

[2024-09-30] [@hugoycj](https://github.com/hugoycj) adds a gradio demo

[2024-09-20] Instructions for datasets [data_preprocess.md](docs/data_preprocess.md)

[2024-09-11] Code for Spann3R

## Installation

1. Clone Spann3R

   ```
   git clone https://github.com/HengyiWang/spann3r.git
   ```

   ```
   cd spann3r
   ```
   
2. Create conda environment

   ```
   conda create -n spann3r python=3.9 cmake=3.14.0
   ```

   Now, activate the conda environment:
   ```
   conda activate spann3r
   ```

   and install more things:

   ```
   conda install pytorch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0 pytorch-cuda=11.8 nvcc-cuda=11.8 ninja -c pytorch -c nvidia -c conda-forge  # use the correct version of cuda for your system
   ```

   ```
   pip install -r requirements.txt
   ```
   
   Open3D has a bug from 0.16.0, so we install the dev version:

   ```
   pip install -U -f https://www.open3d.org/docs/latest/getting_started.html --only-binary open3d open3d
   ```

3. Compile cuda kernels for RoPE

   Now, lets set the CUDA_HOME environment variable with:

   ```
   set CUDA_HOME=%CONDA_PREFIX%
   ```

   and also add clang to our path (using these instructions to find the clang path if it exists: https://stackoverflow.com/a/78316182) and then adding that to the PATH

   ```
   set PATH=%PATH%;<clang-path>
   ```

   Now we change the directory to run `croco/models/curope/setup.py` from it's containing directory (which is important for finding `croco/models/curope/curope.cpp`):

   ```
   cd croco/models/curope/
   ```

   And this is where all our troubles begin:

   ```
   python setup.py build_ext --inplace
   ```

   We currently run into the error:

   ```
   (spann3r) c:\Users\me\Documents\spann3r\croco\models\curope>python setup.py build_ext --inplace
   options (after parsing config files):
   no commands known yet
   options (after parsing command line):
   option dict for 'aliases' command:
   {}
   option dict for 'build_ext' command:
   {'inplace': ('command line', 1)}
   running build_ext
   building 'curope' extension
   creating c:\Users\me\Documents\spann3r\croco\models\curope\build\temp.win-amd64-cpython-39\Release
   Emitting ninja build file c:\Users\me\Documents\spann3r\croco\models\curope\build\temp.win-amd64-cpython-39\Release\build.ninja...
   Compiling objects...
   Allowing ninja to set a default number of workers... (overridable by setting the environment variable MAX_JOBS=N)
   [1/2] C:\Users\me\.conda\envs\spann3r\bin\nvcc --generate-dependencies-with-compile --dependency-output c:\Users\me\Documents\spann3r\croco\models\curope\build\temp.win-amd64-cpython-39\Release\kernels.obj.d -std=c++17 --use-local-env -Xcompiler /MD -Xcompiler /wd4819 -Xcompiler /wd4251 -Xcompiler /wd4244 -Xcompiler /wd4267 -Xcompiler /wd4275 -Xcompiler /wd4018 -Xcompiler /wd4190 -Xcompiler /wd4624 -Xcompiler /wd4067 -Xcompiler /wd4068 -Xcompiler /EHsc -Xcudafe --diag_suppress=base_class_has_different_dll_interface -Xcudafe --diag_suppress=field_without_dll_interface -Xcudafe --diag_suppress=dll_interface_conflict_none_assumed -Xcudafe --diag_suppress=dll_interface_conflict_dllexport_assumed -IC:\Users\me\.conda\envs\spann3r\lib\site-packages\torch\include -IC:\Users\me\.conda\envs\spann3r\lib\site-packages\torch\include\torch\csrc\api\include -IC:\Users\me\.conda\envs\spann3r\lib\site-packages\torch\include\TH -IC:\Users\me\.conda\envs\spann3r\lib\site-packages\torch\include\THC -IC:\Users\me\.conda\envs\spann3r\include -IC:\Users\me\.conda\envs\spann3r\include -IC:\Users\me\.conda\envs\spann3r\Include "-IC:\Program Files\Microsoft Visual Studio\2022\Community\VC\Tools\MSVC\14.36.32532\include" "-IC:\Program Files\Microsoft Visual Studio\2022\Community\VC\Tools\MSVC\14.36.32532\ATLMFC\include" "-IC:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\VS\include" "-IC:\Program Files (x86)\Windows Kits\10\include\10.0.22000.0\ucrt" "-IC:\Program Files (x86)\Windows Kits\10\\include\10.0.22000.0\\um" "-IC:\Program Files (x86)\Windows Kits\10\\include\10.0.22000.0\\shared" "-IC:\Program Files (x86)\Windows Kits\10\\include\10.0.22000.0\\winrt" "-IC:\Program Files (x86)\Windows Kits\10\\include\10.0.22000.0\\cppwinrt" "-IC:\Program Files (x86)\Windows Kits\NETFXSDK\4.8\include\um" -c c:\Users\me\Documents\spann3r\croco\models\curope\kernels.cu -o c:\Users\me\Documents\spann3r\croco\models\curope\build\temp.win-amd64-cpython-39\Release\kernels.obj -D__CUDA_NO_HALF_OPERATORS__ -D__CUDA_NO_HALF_CONVERSIONS__ -D__CUDA_NO_BFLOAT16_CONVERSIONS__ -D__CUDA_NO_HALF2_OPERATORS__ --expt-relaxed-constexpr -O3 --ptxas-options=-v --use_fast_math -gencode arch=compute_50,code=sm_50 -gencode arch=compute_60,code=sm_60 -gencode arch=compute_61,code=sm_61 -gencode arch=compute_70,code=sm_70 -gencode arch=compute_75,code=sm_75 -gencode arch=compute_80,code=sm_80 -gencode arch=compute_86,code=sm_86 -gencode arch=compute_37,code=sm_37 -gencode arch=compute_90,code=sm_90 -gencode arch=compute_37,code=compute_37 -DTORCH_API_INCLUDE_EXTENSION_H -DTORCH_EXTENSION_NAME=curope -D_GLIBCXX_USE_CXX11_ABI=0
   FAILED: c:/Users/me/Documents/spann3r/croco/models/curope/build/temp.win-amd64-cpython-39/Release/kernels.obj
   C:\Users\me\.conda\envs\spann3r\bin\nvcc --generate-dependencies-with-compile --dependency-output c:\Users\me\Documents\spann3r\croco\models\curope\build\temp.win-amd64-cpython-39\Release\kernels.obj.d -std=c++17 --use-local-env -Xcompiler /MD -Xcompiler /wd4819 -Xcompiler /wd4251 -Xcompiler /wd4244 -Xcompiler /wd4267 -Xcompiler /wd4275 -Xcompiler /wd4018 -Xcompiler /wd4190 -Xcompiler /wd4624 -Xcompiler /wd4067 -Xcompiler /wd4068 -Xcompiler /EHsc -Xcudafe --diag_suppress=base_class_has_different_dll_interface -Xcudafe --diag_suppress=field_without_dll_interface -Xcudafe --diag_suppress=dll_interface_conflict_none_assumed -Xcudafe --diag_suppress=dll_interface_conflict_dllexport_assumed -IC:\Users\me\.conda\envs\spann3r\lib\site-packages\torch\include -IC:\Users\me\.conda\envs\spann3r\lib\site-packages\torch\include\torch\csrc\api\include -IC:\Users\me\.conda\envs\spann3r\lib\site-packages\torch\include\TH -IC:\Users\me\.conda\envs\spann3r\lib\site-packages\torch\include\THC -IC:\Users\me\.conda\envs\spann3r\include -IC:\Users\me\.conda\envs\spann3r\include -IC:\Users\me\.conda\envs\spann3r\Include "-IC:\Program Files\Microsoft Visual Studio\2022\Community\VC\Tools\MSVC\14.36.32532\include" "-IC:\Program Files\Microsoft Visual Studio\2022\Community\VC\Tools\MSVC\14.36.32532\ATLMFC\include" "-IC:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\VS\include" "-IC:\Program Files (x86)\Windows Kits\10\include\10.0.22000.0\ucrt" "-IC:\Program Files (x86)\Windows Kits\10\\include\10.0.22000.0\\um" "-IC:\Program Files (x86)\Windows Kits\10\\include\10.0.22000.0\\shared" "-IC:\Program Files (x86)\Windows Kits\10\\include\10.0.22000.0\\winrt" "-IC:\Program Files (x86)\Windows Kits\10\\include\10.0.22000.0\\cppwinrt" "-IC:\Program Files (x86)\Windows Kits\NETFXSDK\4.8\include\um" -c c:\Users\me\Documents\spann3r\croco\models\curope\kernels.cu -o c:\Users\me\Documents\spann3r\croco\models\curope\build\temp.win-amd64-cpython-39\Release\kernels.obj -D__CUDA_NO_HALF_OPERATORS__ -D__CUDA_NO_HALF_CONVERSIONS__ -D__CUDA_NO_BFLOAT16_CONVERSIONS__ -D__CUDA_NO_HALF2_OPERATORS__ --expt-relaxed-constexpr -O3 --ptxas-options=-v --use_fast_math -gencode arch=compute_50,code=sm_50 -gencode arch=compute_60,code=sm_60 -gencode arch=compute_61,code=sm_61 -gencode arch=compute_70,code=sm_70 -gencode arch=compute_75,code=sm_75 -gencode arch=compute_80,code=sm_80 -gencode arch=compute_86,code=sm_86 -gencode arch=compute_37,code=sm_37 -gencode arch=compute_90,code=sm_90 -gencode arch=compute_37,code=compute_37 -DTORCH_API_INCLUDE_EXTENSION_H -DTORCH_EXTENSION_NAME=curope -D_GLIBCXX_USE_CXX11_ABI=0
   C:/Users/me/.conda/envs/spann3r/lib/site-packages/torch/include\c10/util/complex.h(8): fatal error C1083: Cannot open include file: 'thrust/complex.h': No such file or directory
   nvcc warning : The 'compute_35', 'compute_37', 'sm_35', and 'sm_37' architectures are deprecated, and may be removed in a future release (Use -Wno-deprecated-gpu-targets to suppress warning).
   kernels.cu
   [2/2] cl /showIncludes /nologo /O2 /W3 /GL /DNDEBUG /MD /MD /wd4819 /wd4251 /wd4244 /wd4267 /wd4275 /wd4018 /wd4190 /wd4624 /wd4067 /wd4068 /EHsc -IC:\Users\me\.conda\envs\spann3r\lib\site-packages\torch\include -IC:\Users\me\.conda\envs\spann3r\lib\site-packages\torch\include\torch\csrc\api\include -IC:\Users\me\.conda\envs\spann3r\lib\site-packages\torch\include\TH -IC:\Users\me\.conda\envs\spann3r\lib\site-packages\torch\include\THC -IC:\Users\me\.conda\envs\spann3r\include -IC:\Users\me\.conda\envs\spann3r\include -IC:\Users\me\.conda\envs\spann3r\Include "-IC:\Program Files\Microsoft Visual Studio\2022\Community\VC\Tools\MSVC\14.36.32532\include" "-IC:\Program Files\Microsoft Visual Studio\2022\Community\VC\Tools\MSVC\14.36.32532\ATLMFC\include" "-IC:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\VS\include" "-IC:\Program Files (x86)\Windows Kits\10\include\10.0.22000.0\ucrt" "-IC:\Program Files (x86)\Windows Kits\10\\include\10.0.22000.0\\um" "-IC:\Program Files (x86)\Windows Kits\10\\include\10.0.22000.0\\shared" "-IC:\Program Files (x86)\Windows Kits\10\\include\10.0.22000.0\\winrt" "-IC:\Program Files (x86)\Windows Kits\10\\include\10.0.22000.0\\cppwinrt" "-IC:\Program Files (x86)\Windows Kits\NETFXSDK\4.8\include\um" -c c:\Users\me\Documents\spann3r\croco\models\curope\curope.cpp /Foc:\Users\me\Documents\spann3r\croco\models\curope\build\temp.win-amd64-cpython-39\Release\curope.obj -O3 -DTORCH_API_INCLUDE_EXTENSION_H -DTORCH_EXTENSION_NAME=curope -D_GLIBCXX_USE_CXX11_ABI=0 /std:c++17
   cl : Command line warning D9002 : ignoring unknown option '-O3'
   ninja: build stopped: subcommand failed.
   Traceback (most recent call last):
   File "C:\Users\me\.conda\envs\spann3r\lib\site-packages\torch\utils\cpp_extension.py", line 2109, in _run_ninja_build
      subprocess.run(
   File "C:\Users\me\.conda\envs\spann3r\lib\subprocess.py", line 541, in run
      raise CalledProcessError(retcode, process.args,
   subprocess.CalledProcessError: Command '['ninja', '-v']' returned non-zero exit status 1.

   The above exception was the direct cause of the following exception:

   Traceback (most recent call last):
   File "c:\Users\me\Documents\spann3r\croco\models\curope\setup.py", line 25, in <module>
      setup(
   File "C:\Users\me\.conda\envs\spann3r\lib\site-packages\setuptools\__init__.py", line 117, in setup
      return distutils.core.setup(**attrs)
   File "C:\Users\me\.conda\envs\spann3r\lib\site-packages\setuptools\_distutils\core.py", line 184, in setup
      return run_commands(dist)
   File "C:\Users\me\.conda\envs\spann3r\lib\site-packages\setuptools\_distutils\core.py", line 200, in run_commands
      dist.run_commands()
   File "C:\Users\me\.conda\envs\spann3r\lib\site-packages\setuptools\_distutils\dist.py", line 954, in run_commands
      self.run_command(cmd)
   File "C:\Users\me\.conda\envs\spann3r\lib\site-packages\setuptools\dist.py", line 950, in run_command
      super().run_command(command)
   File "C:\Users\me\.conda\envs\spann3r\lib\site-packages\setuptools\_distutils\dist.py", line 973, in run_command
      cmd_obj.run()
   File "C:\Users\me\.conda\envs\spann3r\lib\site-packages\setuptools\command\build_ext.py", line 98, in run
      _build_ext.run(self)
   File "C:\Users\me\.conda\envs\spann3r\lib\site-packages\setuptools\_distutils\command\build_ext.py", line 359, in run
      self.build_extensions()
   File "C:\Users\me\.conda\envs\spann3r\lib\site-packages\torch\utils\cpp_extension.py", line 870, in build_extensions
      build_ext.build_extensions(self)
   File "C:\Users\me\.conda\envs\spann3r\lib\site-packages\setuptools\_distutils\command\build_ext.py", line 476, in build_extensions
      self._build_extensions_serial()
   File "C:\Users\me\.conda\envs\spann3r\lib\site-packages\setuptools\_distutils\command\build_ext.py", line 502, in _build_extensions_serial
      self.build_extension(ext)
   File "C:\Users\me\.conda\envs\spann3r\lib\site-packages\setuptools\command\build_ext.py", line 263, in build_extension
   File "C:\Users\me\.conda\envs\spann3r\lib\site-packages\setuptools\command\build_ext.py", line 263, in build_extension
   File "C:\Users\me\.conda\envs\spann3r\lib\site-packages\setuptools\command\build_ext.py", line 263, in build_extension
      _build_ext.build_extension(self, ext)
   File "C:\Users\me\.conda\envs\spann3r\lib\site-packages\setuptools\command\build_ext.py", line 263, in build_extension
      _build_ext.build_extension(self, ext)
   File "C:\Users\me\.conda\envs\spann3r\lib\site-packages\setuptools\command\build_ext.py", line 263, in build_extension
   File "C:\Users\me\.conda\envs\spann3r\lib\site-packages\setuptools\command\build_ext.py", line 263, in build_extension
   File "C:\Users\me\.conda\envs\spann3r\lib\site-packages\setuptools\command\build_ext.py", line 263, in build_extension
   File "C:\Users\me\.conda\envs\spann3r\lib\site-packages\setuptools\command\build_ext.py", line 263, in build_extension
   File "C:\Users\me\.conda\envs\spann3r\lib\site-packages\setuptools\command\build_ext.py", line 263, in build_extension
   File "C:\Users\me\.conda\envs\spann3r\lib\site-packages\setuptools\command\build_ext.py", line 263, in build_extension
   File "C:\Users\me\.conda\envs\spann3r\lib\site-packages\setuptools\command\build_ext.py", line 263, in build_extension
   File "C:\Users\me\.conda\envs\spann3r\lib\site-packages\setuptools\command\build_ext.py", line 263, in build_extension
      _build_ext.build_extension(self, ext)
   File "C:\Users\me\.conda\envs\spann3r\lib\site-packages\setuptools\_distutils\command\build_ext.py", line 557, in build_extension
   File "C:\Users\me\.conda\envs\spann3r\lib\site-packages\setuptools\command\build_ext.py", line 263, in build_extension
      _build_ext.build_extension(self, ext)
   File "C:\Users\me\.conda\envs\spann3r\lib\site-packages\setuptools\_distutils\command\build_ext.py", line 557, in build_extension
      objects = self.compiler.compile(
      _run_ninja_build(
   File "C:\Users\me\.conda\envs\spann3r\lib\site-packages\torch\utils\cpp_extension.py", line 2125, in _run_ninja_build
      raise RuntimeError(message) from e
   RuntimeError: Error compiling objects for extension
   ```

   Continuing the installation instructions:

   ```
   cd ../../../
   ```

4. Download the DUSt3R checkpoint

   ```
   mkdir checkpoints
   cd checkpoints
   # Download DUSt3R checkpoints
   wget https://download.europe.naverlabs.com/ComputerVision/DUSt3R/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth
   ```

5. Download our [checkpoint](https://drive.google.com/drive/folders/1bqtcVf8lK4VC8LgG-SIGRBECcrFqM7Wy?usp=sharing) and place it under `./checkpoints`

## Demo

1. Download the [example data](https://drive.google.com/drive/folders/1bqtcVf8lK4VC8LgG-SIGRBECcrFqM7Wy?usp=sharing) (2 scenes from [map-free-reloc](https://github.com/nianticlabs/map-free-reloc)) and unzip it as `./examples`

2. Run demo:

   ```
   python demo.py --demo_path ./examples/s00567 --kf_every 10 --vis --vis_cam
   ```

   For visualization `--vis`, it will give you a window to adjust the rendering view. Once you find the view to render, please click `space key` and close the window. The code will then do the rendering of the incremental reconstruction.
   
3. Nerfstudio:

   ```
   # Run demo use --save_ori to save scaled intrinsics for original images
   python demo.py --demo_path ./examples/s00567 --kf_every 10 --vis --vis_cam --save_ori
   
   # Run splatfacto
   ns-train splatfacto --data ./output/demo/s00567 --pipeline.model.camera-optimizer.mode SO3xR3
   
   # Render your results
   ns-render interpolate --load-config [path-to-your-config]/config.yml
   ```

   Note that here you can use `--save_ori` to save the scaled intrinsics into `transform.json` to train NeRF/3D Gaussians with original images.'


## Gradio interface 

We also provide a Gradio interface for a better experience, just run by:

```bash
# For Linux and Windows users (and macOS with Intel??)
python app.py
```

You can specify the `--server_port`, `--share`, `--server_name` arguments to satisfy your needs!


## Training and Evaluation

### Datasets

We use Habitat, ScanNet++, ScanNet, ArkitScenes, Co3D, and BlendedMVS to train our model. Please refer to [data_preprocess.md](docs/data_preprocess.md).

### Train

Please use the following command to train our model:

```
torchrun --nproc_per_node 8 train.py --batch_size 4
```

### Eval

Please use the following command to evaluate our model:

```
python eval.py
```




## Acknowledgement 

Our code, data preprocessing pipeline, and evaluation scripts are based on several awesome repositories:

- [DUSt3R](https://github.com/naver/dust3r)
- [SplaTAM](https://github.com/spla-tam/SplaTAM)
- [NeRFStudio](https://github.com/nerfstudio-project/nerfstudio)
- [MVSNet](https://github.com/YoYo000/MVSNet)
- [NICE-SLAM](https://github.com/cvg/nice-slam)
- [NeuralRGBD](https://github.com/dazinovic/neural-rgbd-surface-reconstruction)
- [SimpleRecon](https://github.com/nianticlabs/simplerecon)

We thank the authors for releasing their code!

The research presented here has been supported by a sponsored research award from Cisco Research and the UCL Centre for Doctoral Training in Foundational AI under UKRI grant number EP/S021566/1. This project made use of time on Tier 2 HPC facility JADE2, funded by EPSRC (EP/T022205/1).

## Citation

If you find our code or paper useful for your research, please consider citing:

```
@article{wang20243d,
  title={3D Reconstruction with Spatial Memory},
  author={Wang, Hengyi and Agapito, Lourdes},
  journal={arXiv preprint arXiv:2408.16061},
  year={2024}
}
```

