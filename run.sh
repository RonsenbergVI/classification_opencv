SOURCE="${BASH_SOURCE[0]}"

DIR="$( cd -P "$( dirname "$SOURCE")" && pwd )/extracts/"

g++ source/main.cpp -lopencv_core -lopencv_ml -lopencv_imgproc -o source/opencv.out

source/opencv.out $DIR
