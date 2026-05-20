# Makefile for Tamarin ML Evaluation Models
# Usage: make all    - Run all models
#        make list   - List all discovered models
#        make clean  - Remove all results folders

# Timeout parameter (adjust as needed)
TIMEOUT = 7200

# Python interpreter
PYTHON = python3

# Tamarin wrapper script
WRAPPER = ./tamarin_wrapper.py

# Auto-discover all .spthy files (excluding those in results folders)
SPTHY_FILES := $(shell find . -name "*.spthy" -not -path "*/results/*" -not -path "*/.*")

# Split files into 8 parts for concurrent processing
TOTAL_FILES := $(words $(SPTHY_FILES))
PART_SIZE := $(shell echo $$(($(TOTAL_FILES) / 8)))
REMAINDER := $(shell echo $$(($(TOTAL_FILES) % 8)))

# Calculate boundaries for each part
END1 := $(shell echo $$(($(PART_SIZE) + ($(REMAINDER) > 0 ? 1 : 0))))
END2 := $(shell echo $$(($(END1) + $(PART_SIZE) + ($(REMAINDER) > 1 ? 1 : 0))))
END3 := $(shell echo $$(($(END2) + $(PART_SIZE) + ($(REMAINDER) > 2 ? 1 : 0))))
END4 := $(shell echo $$(($(END3) + $(PART_SIZE) + ($(REMAINDER) > 3 ? 1 : 0))))
END5 := $(shell echo $$(($(END4) + $(PART_SIZE) + ($(REMAINDER) > 4 ? 1 : 0))))
END6 := $(shell echo $$(($(END5) + $(PART_SIZE) + ($(REMAINDER) > 5 ? 1 : 0))))
END7 := $(shell echo $$(($(END6) + $(PART_SIZE) + ($(REMAINDER) > 6 ? 1 : 0))))

SPTHY_FILES_PART1 := $(wordlist 1,$(END1),$(SPTHY_FILES))
SPTHY_FILES_PART2 := $(wordlist $(shell echo $$(($(END1) + 1))),$(END2),$(SPTHY_FILES))
SPTHY_FILES_PART3 := $(wordlist $(shell echo $$(($(END2) + 1))),$(END3),$(SPTHY_FILES))
SPTHY_FILES_PART4 := $(wordlist $(shell echo $$(($(END3) + 1))),$(END4),$(SPTHY_FILES))
SPTHY_FILES_PART5 := $(wordlist $(shell echo $$(($(END4) + 1))),$(END5),$(SPTHY_FILES))
SPTHY_FILES_PART6 := $(wordlist $(shell echo $$(($(END5) + 1))),$(END6),$(SPTHY_FILES))
SPTHY_FILES_PART7 := $(wordlist $(shell echo $$(($(END6) + 1))),$(END7),$(SPTHY_FILES))
SPTHY_FILES_PART8 := $(wordlist $(shell echo $$(($(END7) + 1))),$(TOTAL_FILES),$(SPTHY_FILES))

# Extract unique directories containing .spthy files
SPTHY_DIRS := $(sort $(dir $(SPTHY_FILES)))

.PHONY: all clean list part1 part2 part3 part4 part5 part6 part7 part8 FORCE

all: $(SPTHY_FILES)

# Pattern rule to process any .spthy file (always runs due to FORCE)
%.spthy: FORCE
	@echo "Processing $@..."
	$(PYTHON) $(WRAPPER) $@ -t $(TIMEOUT) -c 8 --resultfolder=../results/tamarin/$(dir $@)

FORCE:

# Process each part of models
part1: $(SPTHY_FILES_PART1)

part2: $(SPTHY_FILES_PART2)

part3: $(SPTHY_FILES_PART3)

part4: $(SPTHY_FILES_PART4)

part5: $(SPTHY_FILES_PART5)

part6: $(SPTHY_FILES_PART6)

part7: $(SPTHY_FILES_PART7)

part8: $(SPTHY_FILES_PART8)

# List all discovered .spthy files
list:
	@echo "Discovered .spthy files:"
	@$(foreach file,$(SPTHY_FILES),echo "  $(file)";)
	@echo ""
	@echo "Total files: $(TOTAL_FILES)"
	@echo ""
	@echo "Part 1 ($(words $(SPTHY_FILES_PART1)) files):"
	@$(foreach file,$(SPTHY_FILES_PART1),echo "  $(file)";)
	@echo ""
	@echo "Part 2 ($(words $(SPTHY_FILES_PART2)) files):"
	@$(foreach file,$(SPTHY_FILES_PART2),echo "  $(file)";)
	@echo ""
	@echo "Part 3 ($(words $(SPTHY_FILES_PART3)) files):"
	@$(foreach file,$(SPTHY_FILES_PART3),echo "  $(file)";)
	@echo ""
	@echo "Part 4 ($(words $(SPTHY_FILES_PART4)) files):"
	@$(foreach file,$(SPTHY_FILES_PART4),echo "  $(file)";)
	@echo ""
	@echo "Part 5 ($(words $(SPTHY_FILES_PART5)) files):"
	@$(foreach file,$(SPTHY_FILES_PART5),echo "  $(file)";)
	@echo ""
	@echo "Part 6 ($(words $(SPTHY_FILES_PART6)) files):"
	@$(foreach file,$(SPTHY_FILES_PART6),echo "  $(file)";)
	@echo ""
	@echo "Part 7 ($(words $(SPTHY_FILES_PART7)) files):"
	@$(foreach file,$(SPTHY_FILES_PART7),echo "  $(file)";)
	@echo ""
	@echo "Part 8 ($(words $(SPTHY_FILES_PART8)) files):"
	@$(foreach file,$(SPTHY_FILES_PART8),echo "  $(file)";)

# Clean all results folders
clean:
	@echo "Removing results folders..."
	@rm -rf ../results/tamarin
	@echo "Done."
