##
#
# Author: Nikolaus Mayer, 2015 (mayern@cs.uni-freiburg.de)
#
##

## Where to look for includes (default is 'here')
INCLUDE_DIRS ?= -I. 

CXX ?= /usr/bin/g++

## Compiler flags; extended in 'debug'/'release' rules
CXXFLAGS ?= -W -Wall -Wextra -fPIC

## Linker flags
LDFLAGS ?= 

## Default name for the built executable
TARGET = FlowVisualization.so

## Every *.c/*.cc/*.cpp file is a source file
SRCS = $(wildcard *.c *.cc *.cpp)

## Build a *.o object file for every source file
OBJS = $(addsuffix .o, $(basename $(SRCS)))


## Tell make that e.g. 'make clean' is not supposed to create a file 'clean'
##
## "Why is it called 'phony'?" -- because it's not a real target. That is, 
## the target name isn't a file that is produced by the commands of that target.
.PHONY: all clean debug release


## Default is release build mode
all: release
	
## When in debug mode, don't optimize, and create debug symbols
debug: CXXFLAGS += -O0 -g
debug: $(TARGET)
	
## When in release mode, optimize
release: CXXFLAGS += -O3
release: $(TARGET)

## Remove built object files and the main executable
clean:
	$(info ... deleting built object files and executable  ...)
	-rm *.o $(TARGET)

## The main executable depends on all object files of all source files
%.so: $(OBJS)
	$(info ... compiling shared object $@ ...)
	$(CXX) $^ -shared -o $(LDFLAGS) $@

## Every object file depends on its source and the makefile itself
%.o: %.$(CXX) Makefile
	$(info ... compiling $@ ...)
	$(CXX) $(CXXFLAGS) $(INCLUDE_DIRS) -c $< -o $@

%.o: %.cpp Makefile
	$(info ... compiling $@ ...)
	$(CXX) $(CXXFLAGS) $(INCLUDE_DIRS) -c $< -o $@



