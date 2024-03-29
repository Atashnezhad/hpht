# make a parameter to save date and time of execution
DATE := $(shell date +%Y-%m-%d_%H-%M-%S)

# make a new requirements.txt file
.PHONY: new-requirements
new-requirements:
	@touch requirements.txt


# update pip
.PHONY: update-pip
update-pip:
	@echo "Updating pip..."
	@pip install --upgrade pip

# install required packages
.PHONY: install_requirements
install_requirements:
	@make update-pip
	@echo "Installing required packages..."
	@pip install -r requirements.txt


# get all pip list packages and save to requirements.txt
.PHONY: freeze
freeze:
	@echo "Freezing requirements..."
	@pip freeze > requirements.txt

# make automated git
# pass a customized argument to the make command
# include the date and time into the message
.PHONY: git-automated
git-automated:
	# remove the .idea folder
	# remember that another option is to add it to the
	# .gitignore file so it is not include in the git
	git add .
	git commit -m "$(message)"
	git push