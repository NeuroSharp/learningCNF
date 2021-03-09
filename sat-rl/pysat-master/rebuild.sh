# Clean up (Move the cleaning stuff to the makefile's of individual solvers)
python setup.py clean
rm -r build

rm ./solvers/minisat22/libminisat22.a

rm ./solvers/glucose30/libglucose30.a
find ./solvers/glucose30 -name "*.o" -type f -delete

rm ./solvers/lingeling/liblingeling.a

# rm ./solvers/sharpSAT/libsharpSAT.a
# rm -rf ./solvers/sharpSAT/build
# find . -name "*.o" -type f -delete

# Build
python setup.py install --force
