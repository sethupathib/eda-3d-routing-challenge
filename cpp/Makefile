CXX ?= g++
CXXFLAGS ?= -O3 -std=c++17 -pthread -pipe
TARGET := m3d_route

.PHONY: all clean

all: $(TARGET)

$(TARGET): m3d_route.cpp
	$(CXX) $(CXXFLAGS) -o $@ $<

clean:
	rm -f $(TARGET)
