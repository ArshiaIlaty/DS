import json
import sys
import os

def convert_py_to_ipynb(input_file, output_file=None):
    """
    Convert a .py file containing Jupyter notebook JSON to a proper .ipynb file.
    
    Args:
        input_file (str): Path to the input .py file with notebook JSON content
        output_file (str, optional): Path for the output .ipynb file. If not provided,
                                    will use the input filename with .ipynb extension.
    """
    try:
        # Read the content of the .py file
        with open(input_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Parse the JSON content
        notebook_json = json.loads(content)
        
        # Generate output filename if not provided
        if output_file is None:
            base_name = os.path.splitext(input_file)[0]
            output_file = f"{base_name}.ipynb"
        
        # Write the JSON to a proper .ipynb file
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(notebook_json, f, indent=2)
        
        print(f"Successfully converted {input_file} to {output_file}")
        return True
    
    except json.JSONDecodeError:
        print(f"Error: The file {input_file} does not contain valid JSON data.")
        return False
    except Exception as e:
        print(f"Error during conversion: {str(e)}")
        return False

if __name__ == "__main__":
    # If run as a script, take command line arguments
    if len(sys.argv) < 2:
        print("Usage: python convert_py_to_ipynb.py input_file.py [output_file.ipynb]")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else None
    
    success = convert_py_to_ipynb(input_file, output_file)
    sys.exit(0 if success else 1)