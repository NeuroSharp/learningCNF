### Makefile ---

## Author: aign
## Keywords:

# main parameters
CC       := cc
CFLAGS   := -fPIC -Wall -Wno-deprecated -fno-strict-aliasing
LFLAGS   := -DNLGLOG -DNDEBUG -DNCHKSOL -DNLGLYALSAT -DNLGLFILES -DNLGLDEMA
VERSION  := -DVERSION=\"PYSAT\"
INCLUDES := -I.
LIBS     := -L.
SOURCES  := lglib.c lglopts.c druplig.c
OBJECTS  := $(SOURCES:.c=.o)
TRGT     := lingeling

# gather compiler options
CARGS := $(CFLAGS) $(INCLUDES) $(LFLAGS) $(VERSION)

# targets
.PHONY: all clean lib

all: lib

lib: $(OBJECTS)
	@-ar cr lib$(TRGT).a $(OBJECTS)

%.o: %.c
	$(CC) $(CARGS) -c $< -o $@

clean:
	@-rm $(OBJECTS)

allclean:
	@-find . -name '*.o' -delete
	@-find . -name '*.a' -delete

#-------------------------------------------------------------------------------
