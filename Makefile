BUILD_DIR=build
OUT_DIR=out

all:
	+make -C $(BUILD_DIR)

doc:
	mkdir -p out
	+make -C $(BUILD_DIR) $@

distclean:
	$(RM) $(DIRS) $(BUILD_DIR)
	$(RM) $(OUT_DIR)

%:
	+make -C $(BUILD_DIR) $@

.PHONY: doc
