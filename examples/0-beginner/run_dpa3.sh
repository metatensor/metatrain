export METATENSOR_DEBUG_EXTENSIONS_LOADING=1

# mtt train options.yaml

package_dir=$(python -c "import site; print(site.getsitepackages()[0])")
cp $package_dir/deepmd/lib/*.so extensions/deepmd/lib/
cp $package_dir/deepmd_kit.libs/*.so* extensions/deepmd/lib/

mtt eval model.pt eval.yaml -e extensions/