# Compilation

- Get meson, specifically: `pip install meson==0.61.3`.
  This avoids an issue in compiling capnprot for seticore, that occurs with v0.64.0.

- Get BLADE: `git clone https://github.com/MydonSolutions/blade.git && cd blade && git checkout blade-cli-vla`

- Pull all submodules (recursively): `git submodule update --recursive --init`

- Setup the build folder: `CXX=g++-10 meson build -Dseticore:werror=false && cd build`

- Compile: `ninja`

# Primary Sky-Data Test

The ATA recorded a pair of RAW stems, one with phase center on Mars and another offset from Mars (+0.1 hours in Right-Ascension, and +0.001 degrees in Declination). The beamforming of these is qualitatively indicative of effective beamforming.

Retrieve the data
```
rsync -avW rsync://blpd18.ssl.berkeley.edu/datax/guppi_59856_57567_103605834_mars_0001.bfr5 .
rsync -avW rsync://blpd18.ssl.berkeley.edu/datax/guppi_59856_57567_103605834_mars_0001.0000.raw .
```

```
rsync -avW rsync://blpd18.ssl.berkeley.edu/datax/guppi_59856_58696_103674743_mars_off_0001.0000.raw .
rsync -avW rsync://blpd18.ssl.berkeley.edu/datax/guppi_59856_58696_103674743_mars_off_0001.bfr5 .
```

These BFR5 files were produced by `bfr5_gen.jl` from this [repo](https://github.com/MydonSolutions/ata_bfr5_genie) (some local changes were made to produce the appropriate beam coordinates for the _off dataset).

Then beamform based on a RAW-BFR5 input pair:

`./build/apps/blade-cli/blade-cli --input-type CI8 --output-type F32 -t ATA -m B -c 128 -T 32 -C 1 -N 1 ./guppi_59856_58696_103674743_mars_off_0001.0000.raw ./guppi_59856_58696_103674743_mars_off_0001.bfr5 ./guppi_59856_58696_103674743_mars_off_output`

`watutil -p ank -b 8431.2 -e 8431.4` was used to produce the following plots of the output filterbank files.

![mars_on reference output](./img/guppi_59856_57567_103605834_mars_0001.png)

![mars_off reference output](./img/guppi_59856_58696_103674743_mars_off_0001.png)

# Python Beamforming for Comparison

Beamform in Python (the output is in the same directory of the raw file and suffixed with -beam{:04d}.0000.raw):

`./tests/apps/blade-cli/beamform.py synthetic_test_rand.bfr5 synthetic_test_rand.0000.raw -u 1`

Compare in Julia:

`./tests/apps/blade-cli/compare_raw.jl synthetic_test_rand_bladeout.0000.raw synthetic_test_rand-beam000.0000.raw `

# Manual Synthesized BLADE-cli test

The BLADE-cli is currently only verified to execute and not crash.

It ingests both a RAW file and a BFR5 file. These are synthesized as follows.

- **GUPPI RAW Synthesis**
	- Install setigen: `pip install setigen`
	- `python ./synthesize_guppi_raw.py`

- **BFR5 Synthesis**
	- Install BeamformerRecipe.jl: `julia> using Pkg; Pkg.add(url="https://github.com/david-macmahon/BeamformerRecipes.jl")`
	- `julia ./synthesize_bf_recipe.jl`

The command run and its output are:

```
root@e4d54b186562:/blade/build/apps/blade-cli# ./blade-cli -t ATA -m B /blade/tests/apps/blade-cli/synthesized_input /blade/tests/apps/blade-cli/synthesized_input.bfr5 
BLADE [INFO]  | Input GUPPI RAW File Path: /blade/tests/apps/blade-cli/synthesized_input
BLADE [INFO]  | Input BFR5 File Path: /blade/tests/apps/blade-cli/synthesized_input.bfr5
BLADE [INFO]  | Telescope: 0
BLADE [INFO]  | Mode: 0
BLADE [INFO]  | Fine-time: 32
BLADE [INFO]  | Coarse-channels: 32
BLADE [INFO]  | ===== GUPPI Reader Module Configuration
BLADE [INFO]  | Input File Path: /blade/tests/apps/blade-cli/synthesized_input
BLADE [INFO]  | Datashape: [20, 128, 8192, 2, CI4] (41943040 bytes)
BLADE [INFO]  | Read 41943040 Elements, Dimension Lengths: [20, 32, 32768, 2] (83886080 bytes)
BLADE [INFO]  | ===== BFR5 Reader Module Configuration
BLADE [INFO]  | Input File Path: /blade/tests/apps/blade-cli/synthesized_input.bfr5
BLADE [INFO]  | Dim_info: [20, 128, 16384, 2] - 8 Beams
BLADE [WARN]  | Sub-band processing of the coarse-channels (32/128) is incompletely implemented: only the first sub-band is processed.
BLADE [INFO]  | 

Welcome to BLADE (Breakthrough Listen Accelerated DSP Engine)!
Version 0.6.0 | Build Type: release
                   .-.
    .-""`""-.    |(0 0)
 _/`oOoOoOoOo`\_ \ \-/
'.-=-=-=-=-=-=-.' \/ \
  `-=.=-.-=.=-'    \ /\
     ^  ^  ^       _H_ \
            
BLADE [INFO]  | Instantiating new runner.
BLADE [DEBUG] | Initializing new worker.
BLADE [DEBUG] | Initializing ATA Pipeline Mode B.
BLADE [DEBUG] | Instantiating input cast from I8 to CF32.
BLADE [INFO]  | ===== Cast Module Configuration
BLADE [INFO]  | Input Size: 41943040
BLADE [DEBUG] | Input is empty, allocating 41943040 elements
BLADE [DEBUG] | Instantiating channelizer with rate 1024.
BLADE [INFO]  | ===== Channelizer Module Configuration
BLADE [INFO]  | Number of Beams: 1
BLADE [INFO]  | Number of Antennas: 20
BLADE [INFO]  | Number of Frequency Channels: 32
BLADE [INFO]  | Number of Time Samples: 32768
BLADE [INFO]  | Number of Polarizations: 2
BLADE [INFO]  | Channelizer Rate: 1024
BLADE [INFO]  | FFT Backend: cuFFT
BLADE [DEBUG] | Instantiating phasor module.
BLADE [INFO]  | ===== Phasor Module Configuration
BLADE [INFO]  | Number of Beams: 8
BLADE [INFO]  | Number of Antennas: 20
BLADE [INFO]  | Number of Frequency Channels: 32768
BLADE [INFO]  | Number of Polarizations: 2
BLADE [INFO]  | RF Frequency (Hz): 1.9762626e-317
BLADE [INFO]  | Channel Bandwidth (Hz): 4.24399170841135e-307
BLADE [INFO]  | Total Bandwidth (Hz): 1.358077346691632e-305
BLADE [INFO]  | Frequency Start Index: 0
BLADE [INFO]  | Reference Antenna Index: 0
BLADE [INFO]  | Array Reference Position (LON, LAT, ALT): (122683.0, -382976.0, 2124.0)
BLADE [INFO]  | Boresight Coordinate (RA, DEC): (0.4552384158732863, 0.16014320814891514)
BLADE [INFO]  | ECEF Antenna Positions (X, Y, Z):
BLADE [INFO]  |     0: (0.6906091426944351, 0.9378720547546184, 0.8828087038225728)
BLADE [INFO]  |     1: (0.3642013618928833, 0.2981249317612238, 0.47883810781135017)
BLADE [INFO]  |     2: (0.11960577624853652, 0.274959193846367, 0.5040632572900199)
BLADE [INFO]  |     3: (0.9219645234195122, 0.557439580277528, 0.6475945198741518)
BLADE [INFO]  |     4: (0.6388436442226182, 0.439751047417997, 0.6511178675037766)
BLADE [INFO]  |     5: (0.8184174399987494, 0.6923013836548134, 0.14497581601590426)
BLADE [INFO]  |     6: (0.7281172973444721, 0.055944448107494105, 0.9079319562351342)
BLADE [INFO]  |     7: (0.07328162868452714, 0.6313600195496789, 0.48097445586991683)
BLADE [INFO]  |     8: (0.4505988530928463, 0.14805125571288824, 0.8104381850017377)
BLADE [INFO]  |     9: (0.800033634667316, 0.38262931925538557, 0.5351099944362997)
BLADE [INFO]  |     10: (0.5226541664715968, 0.07318195487206669, 0.7667234567174142)
BLADE [INFO]  |     11: (0.033513735900588815, 0.06860031879407902, 0.5028610524640852)
BLADE [INFO]  |     12: (0.7152034906382626, 0.23586012389791255, 0.8647869107335733)
BLADE [INFO]  |     13: (0.9525564121043727, 0.8376470732353919, 0.6321466252922635)
BLADE [INFO]  |     14: (0.9431925270393446, 0.6762389847632453, 0.6027221056268577)
BLADE [INFO]  |     15: (0.2376622852712117, 0.3984230622844659, 0.3368008447262628)
BLADE [INFO]  |     16: (0.11238247127883538, 0.3333952991307362, 0.8668041913376953)
BLADE [INFO]  |     17: (0.9963456838639454, 0.05649038933060291, 0.5140203912162432)
BLADE [INFO]  |     18: (0.3203122903051797, 0.8620745812626635, 0.8252952174558081)
BLADE [INFO]  |     19: (0.15240264464224496, 0.8863954866124368, 0.5615488177250053)
BLADE [INFO]  | Beam Coordinates (RA, DEC):
BLADE [INFO]  |     0: (0.4552384158732863, 0.16014320814891514)
BLADE [INFO]  |     1: (0.5476424498276177, 0.5011358617291459)
BLADE [INFO]  |     2: (0.7733535276924052, 0.018743665453639813)
BLADE [INFO]  |     3: (0.9405848223512736, 0.8601828553599953)
BLADE [INFO]  |     4: (0.02964765308691042, 0.6556360448565952)
BLADE [INFO]  |     5: (0.74694291453392, 0.7746656838366666)
BLADE [INFO]  |     6: (0.7468008914093891, 0.7817315740767116)
BLADE [INFO]  |     7: (0.9766699015845924, 0.5553797706980106)
BLADE [DEBUG] | Instantiating beamformer module.
BLADE [INFO]  | ===== Beamformer Module Configuration
BLADE [INFO]  | Number of Beams: 8
BLADE [INFO]  | Number of Antennas: 20
BLADE [INFO]  | Number of Frequency Channels: 32768
BLADE [INFO]  | Number of Time Samples: 32
BLADE [INFO]  | Number of Polarizations: 2
BLADE [INFO]  | Enable Incoherent Beam: NO
BLADE [INFO]  | Enable Incoherent Beam Square Root: NO
BLADE [INFO]  | Allocated Runner output buffer 0: 16777216 (134217728 bytes)
BLADE [DEBUG] | Frame Julian Date: inf
BLADE [DEBUG] | Frame DUT1: 0.0
BLADE [DEBUG] | Caching kernels ahead of CUDA Graph instantiation.
BLADE [DEBUG] | Creating CUDA Graph.
BLADE [vector.hh@24] [FATAL] | Failed to deallocate host memory.
BLADE [vector.hh@24] [FATAL] | Failed to deallocate host memory.
BLADE [vector.hh@26] [FATAL] | Failed to deallocate CUDA memory.
BLADE [pipeline.cc@22] [FATAL] | Failed to synchronize stream: driver shutting down
BLADE [vector.hh@26] [FATAL] | Failed to deallocate CUDA memory.
BLADE [vector.hh@65] [FATAL] | Failed to deallocate CUDA memory.
BLADE [vector.hh@26] [FATAL] | Failed to deallocate CUDA memory.
BLADE [vector.hh@26] [FATAL] | Failed to deallocate CUDA memory.
BLADE [vector.hh@65] [FATAL] | Failed to deallocate CUDA memory.
BLADE [vector.hh@24] [FATAL] | Failed to deallocate host memory.
BLADE [vector.hh@26] [FATAL] | Failed to deallocate CUDA memory.
```