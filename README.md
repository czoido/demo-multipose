```
pip install conan
conan profile detect
conan install . --build=missing

# configure
# for macOS, Linux:
cmake --preset=conan-release
# for Windows:
cmake --preset=conan-default

# build
cmake --build --preset=conan-release
```
