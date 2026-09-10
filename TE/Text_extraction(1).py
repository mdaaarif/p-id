import sys
import os


sys.path.insert(
    0,
    os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
        )
    )
)


from config import config

from src.text_extraction import TextExtractor



def main():


    print("="*70)
    print("P&ID TEXT EXTRACTION")
    print("="*70)


    extractor = TextExtractor(config)


    success = extractor.run()


    if success:
        print("\n✓ COMPLETE")
        return 0

    else:
        print("\n❌ FAILED")
        return 1



if __name__ == "__main__":

    sys.exit(main())