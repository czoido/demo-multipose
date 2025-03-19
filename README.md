```
pip install conan
conan profile detect
conan install . --build=missing
cmake --preset=conan-release
cmake --build --preset=conan-release
```
