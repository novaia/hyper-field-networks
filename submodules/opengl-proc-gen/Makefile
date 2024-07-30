CC = gcc
CFLAGS = -std=c99 -c
LDFLAGS = -lglfw -lglad -lm -lpng
SRCDIR = ./csrc
BUILDDIR = ./build

SRCS = $(shell find $(SRCDIR) -name "*.c")
HDRS = $(shell find $(SRCDIR) -name "*.h")
OBJS = $(patsubst $(SRCDIR)/%.c,$(BUILDDIR)/%.o,$(SRCS))

TARGET = $(BUILDDIR)/pcg

.PHONY: all clean

all: shaders_to_header $(TARGET)

$(TARGET): $(OBJS)
	$(CC) $(OBJS) $(LDFLAGS) -o $(TARGET)

$(BUILDDIR)/%.o: $(SRCDIR)/%.c | $(BUILDDIR)
	$(CC) $(CFLAGS) $< -o $@

$(BUILDDIR):
	mkdir -p $(BUILDDIR)

shaders_to_header:
	bash shaders_to_header.sh

clean:
	rm -rf $(BUILDDIR)

run: all
	./$(TARGET)
