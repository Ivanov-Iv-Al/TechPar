CXX = g++
CXXFLAGS = -O2 -fopenmp -march=native -std=c++17
TARGET = cg

all: $(TARGET)

$(TARGET): main.cpp cg.cpp cg.h
	$(CXX) $(CXXFLAGS) -o $@ main.cpp cg.cpp

run: $(TARGET)
	./$(TARGET) 1000 2000

clean:
	rm -f $(TARGET) results.csv *.png