CC = gcc
CFLAGS = -std=c99 -c
LDFLAGS = -lglfw -lglad -lm -lpng

SRC_DIR = src
BUILD_DIR = build

SOURCES = $(SRC_DIR)/main.c $(SRC_DIR)/file_io.c $(SRC_DIR)/vector_matrix_math.c $(SRC_DIR)/rendering.c
OBJECTS = $(patsubst $(SRC_DIR)/%.c,$(BUILD_DIR)/%.o,$(SOURCES))

TARGET = $(BUILD_DIR)/pcg

all: $(BUILD_DIR) shaders $(TARGET)

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

shaders:
	bash shaders_to_header.sh

$(BUILD_DIR)/%.o: $(SRC_DIR)/%.c
	$(CC) $(CFLAGS) $< -o $@

$(TARGET): $(OBJECTS)
	$(CC) $^ $(LDFLAGS) -o $@

clean:
	rm -rf $(BUILD_DIR) $(TARGET)

.PHONY: all shaders clean

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
