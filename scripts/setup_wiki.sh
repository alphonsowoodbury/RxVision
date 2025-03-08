#!/bin/bash

# Ensure we're in the project root
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR/.."

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${GREEN}Setting up RxVision Wiki...${NC}\n"

# 1. Add wiki remote
echo -e "${YELLOW}Adding wiki remote...${NC}"
git remote remove wiki 2>/dev/null || true
git remote add wiki https://github.com/a-woodbury/RxVision.wiki.git
echo -e "Wiki remote added:\n"
git remote -v | grep wiki

# 2. Ensure wiki directory exists
echo -e "\n${YELLOW}Ensuring wiki directory structure...${NC}"
mkdir -p docs/wiki

# 3. Create post-commit hook for auto-pushing
echo -e "\n${YELLOW}Setting up git hook...${NC}"
HOOK_PATH=".git/hooks/post-commit"
cat > "$HOOK_PATH" << 'EOF'
#!/bin/bash

# Check if docs/wiki was modified in the last commit
if git diff-tree -r --name-only HEAD@{1} HEAD | grep -q "^docs/wiki/"
then
    echo "Wiki changes detected, pushing to wiki repository..."
    git subtree push --prefix=docs/wiki wiki main
fi
EOF
chmod +x "$HOOK_PATH"

# 4. Initial push if content exists
if [ -f "docs/wiki/Home.md" ]; then
    echo -e "\n${YELLOW}Pushing existing wiki content...${NC}"
    git subtree push --prefix=docs/wiki wiki main
fi

echo -e "\n${GREEN}Wiki setup complete!${NC}"
echo -e "\nTo update the wiki in the future:"
echo "1. Make changes in docs/wiki/"
echo "2. Commit your changes"
echo "3. The post-commit hook will automatically push to the wiki"
echo -e "\nOr manually push with: git subtree push --prefix=docs/wiki wiki main" 