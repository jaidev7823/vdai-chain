import requests
from bs4 import BeautifulSoup
import os
from urllib.parse import urljoin, urlparse

def get_all_page_links(base_url):
    """Get all links from base URL that end with '/'"""
    try:
        response = requests.get(base_url, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')
        
        links = set()
        for link in soup.find_all('a', href=True):
            href = link['href']
            full_url = urljoin(base_url, href)
            
            # Only include links that start with base_url and end with '/'
            if full_url.startswith(base_url) and full_url.endswith('/'):
                links.add(full_url)
        
        return sorted(links)
    except Exception as e:
        print(f"Error fetching base URL: {e}")
        return []

def is_likely_function(text):
    """Check if text looks like a function/method/property name"""
    if not text:
        return False
    
    # Explicit function indicators
    if '()' in text or '[' in text or ']' in text:
        return True
    
    # Contains a dot (method or property access like obj.method or obj.property)
    if '.' in text:
        return True
    
    # camelCase pattern (lowercase followed by uppercase, like getElementById)
    has_camel_case = False
    for i in range(len(text) - 1):
        if text[i].islower() and text[i+1].isupper():
            has_camel_case = True
            break
    
    if has_camel_case:
        return True
    
    # Starts with lowercase letter (typical for functions/properties in JS)
    if text and text[0].islower() and any(c.isalpha() for c in text):
        return True
    
    return False

def extract_functions(soup):
    """Extract function information from the page"""
    functions = []
    
    # Find all h3 tags that could contain function names
    h3_tags = soup.find_all('h3')
    
    for h3 in h3_tags:
        h3_text = h3.get_text(strip=True)
        
        # Check if this h3 contains something that looks like a function
        if not is_likely_function(h3_text):
            continue
        
        function_name = h3_text
        description = "None"
        parameters = []
        variants = []
        
        # Look for description and variants in following siblings
        current = h3.find_next_sibling()
        while current:
            # Check if we've hit another h3 (next function)
            if current.name == 'h3':
                break
            
            # Look for h4 with "Description"
            if current.name == 'h4' and 'description' in current.get_text(strip=True).lower():
                desc_elem = current.find_next_sibling()
                if desc_elem:
                    description = desc_elem.get_text(strip=True)
            
            # Look for function variants in <p> tags with <code> tags
            if current.name == 'p':
                code_tags = current.find_all('code')
                for code in code_tags:
                    code_text = code.get_text(strip=True)
                    # Check if code content looks like a function variant
                    if is_likely_function(code_text) and code_text != function_name:
                        variants.append(code_text)
            
            # Look for parameter table
            if current.name == 'table':
                tbody = current.find('tbody')
                if tbody:
                    rows = tbody.find_all('tr')
                    for row in rows:
                        cols = row.find_all('td')
                        if len(cols) >= 3:
                            param_name = cols[0].get_text(strip=True)
                            param_type = cols[1].get_text(strip=True)
                            param_desc = cols[2].get_text(strip=True)
                            parameters.append({
                                'name': param_name,
                                'type': param_type,
                                'description': param_desc
                            })
            
            current = current.find_next_sibling()
        
        # Add main function
        functions.append({
            'name': function_name,
            'description': description,
            'parameters': parameters
        })
        
        # Add variants as separate functions with same description and parameters
        for variant in variants:
            functions.append({
                'name': variant,
                'description': description,
                'parameters': parameters
            })
    
    return functions

def format_output(functions):
    """Format functions into the required output format"""
    output = []
    
    for func in functions:
        output.append(f"Function: {func['name']}")
        output.append(f"Description: {func['description']}")
        
        if func['parameters']:
            params_list = []
            for param in func['parameters']:
                params_list.append(f"{param['name']}|{param['type']}|{param['description']}")
            output.append(f"Parameters: {', '.join(params_list)}")
        else:
            output.append("Parameters: None")
        
        output.append("")  # Empty line between functions
    
    return '\n'.join(output)

def get_slug_from_url(url):
    """Extract slug from URL for filename"""
    parsed = urlparse(url)
    path_parts = [p for p in parsed.path.split('/') if p]
    if path_parts:
        return path_parts[-1]
    return 'index'

def scrape_page(url, output_dir):
    """Scrape a single page and save to file"""
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Extract functions
        functions = extract_functions(soup)
        
        if functions:
            # Format output
            output = format_output(functions)
            
            # Get filename from URL
            slug = get_slug_from_url(url)
            filename = os.path.join(output_dir, f"{slug}.txt")
            
            # Save to file
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(output)
            
            print(f"saved {slug}.txt")
        else:
            slug = get_slug_from_url(url)
            print(f"fail {slug}.txt (no functions found)")
            
    except Exception as e:
        slug = get_slug_from_url(url)
        print(f"fail {slug}.txt")

def main():
    base_url = "https://ppro-scripting.docsforadobe.dev/"
    output_dir = "docs_txt"
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all page links
    print("Fetching page links...")
    page_links = get_all_page_links(base_url)
    
    if not page_links:
        print("No pages found to scrape")
        return
    
    print(f"Found {len(page_links)} pages to scrape\n")
    
    # Scrape each page
    for url in page_links:
        scrape_page(url, output_dir)

if __name__ == "__main__":
    main()