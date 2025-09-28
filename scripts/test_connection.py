#!/usr/bin/env python3
"""
Simple connectivity test for NIH FTP server

Tests basic connection to the NIH RxImage dataset FTP server
without requiring additional dependencies.
"""

def test_ftp_connection():
    """Test basic FTP connection to NIH server"""
    print("Testing FTP connection to NIH RxImage dataset server...")

    try:
        from ftplib import FTP
        import sys

        with FTP('lhcftp.nlm.nih.gov') as ftp:
            ftp.login()
            print("   Connected to NIH FTP server")

            # Navigate to Pills dataset
            ftp.cwd('Open-Access-Datasets/Pills/')
            print("   Navigated to Pills dataset directory")

            # List some directories
            dirs = ftp.nlst()
            pill_dirs = [d for d in dirs if d.startswith('PillProject')]
            print(f"   Found {len(pill_dirs)} PillProject directories")

            # Check for directory file
            files = ftp.nlst()
            if 'directory_of_images.txt' in files:
                print("   Directory index file found")

                # Get file size
                ftp.voidcmd('TYPE I')  # Binary mode
                size = ftp.size('directory_of_images.txt')
                print(f"   Directory file size: {size:,} bytes")

            else:
                print("   Warning: Directory index file not found")

            print("\nFTP connection test: SUCCESS")
            print("The NIH RxImage dataset is accessible.")
            return True

    except ImportError:
        print("   Error: ftplib not available")
        return False
    except Exception as e:
        print(f"   Error: {e}")
        print("\nFTP connection test: FAILED")
        print("Please check your internet connection and try again.")
        return False

def test_dependencies():
    """Test if required dependencies are available"""
    print("\nTesting required dependencies...")

    dependencies = {
        'pandas': 'pip install pandas',
        'numpy': 'pip install numpy',
        'PIL': 'pip install Pillow',
        'pathlib': 'Built-in (Python 3.4+)',
        'ftplib': 'Built-in',
        'tqdm': 'pip install tqdm'
    }

    missing = []

    for dep, install_cmd in dependencies.items():
        try:
            if dep == 'PIL':
                import PIL
            else:
                __import__(dep)
            print(f"   {dep}: Available")
        except ImportError:
            print(f"   {dep}: Missing - {install_cmd}")
            missing.append(dep)

    # Test optional dependencies
    optional_deps = {
        'rawpy': 'pip install rawpy (for Canon RAW image support)',
        'imageio': 'pip install imageio (for advanced image processing)'
    }

    print("\nOptional dependencies (required for data acquisition):")
    for dep, install_cmd in optional_deps.items():
        try:
            __import__(dep)
            print(f"   {dep}: Available")
        except ImportError:
            print(f"   {dep}: Missing - {install_cmd}")
            missing.append(dep)

    if missing:
        print(f"\nMissing dependencies: {', '.join(missing)}")
        print("Install with: pip install -r requirements.txt")
        return False
    else:
        print("\nAll dependencies available!")
        return True

if __name__ == "__main__":
    print("RxVision25 Data Acquisition - Connection Test")
    print("=" * 50)

    # Test dependencies
    deps_ok = test_dependencies()

    print("\n" + "=" * 50)

    # Test FTP connection
    ftp_ok = test_ftp_connection()

    print("\n" + "=" * 50)

    if ftp_ok and deps_ok:
        print("READY: You can now download the NIH RxImage dataset!")
        print("\nNext steps:")
        print("1. python scripts/download_data.py --sample    # Download sample dataset")
        print("2. python scripts/download_data.py --full      # Download full dataset")
        print("3. jupyter notebook notebooks/01_data_exploration_preprocessing.ipynb")
    elif ftp_ok:
        print("FTP connection works, but some dependencies are missing.")
        print("Install missing dependencies with: pip install -r requirements.txt")
    else:
        print("Connection test failed. Please check your internet connection.")