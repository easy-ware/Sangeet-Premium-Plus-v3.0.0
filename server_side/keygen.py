import secrets
import sys
from typing import Union # Import Union for type hint

def generate_secure_hex_key(num_bytes: int) -> str:
    """
    Generates a cryptographically secure hexadecimal token (secret key).

    Args:
        num_bytes: The number of random bytes to generate for the key.
                   Must be an integer. A value of 32 or higher (resulting in
                   a 64+ character hex string) is recommended for security.

    Returns:
        A hexadecimal string representation of the generated random bytes.
        The string will be twice as long as num_bytes (e.g., 32 bytes -> 64 hex chars).

    Raises:
        TypeError: If num_bytes is not an integer.
        ValueError: If num_bytes is less than 32 (enforcing minimum security).
    """
    # --- Input Type Validation ---
    # Use isinstance() for type checking, it's generally more robust than type() ==
    if not isinstance(num_bytes, int):
        raise TypeError(f"Input 'num_bytes' must be an integer, got {type(num_bytes).__name__}.")

    # --- Input Value Validation ---
    # Enforce a minimum number of bytes for security. 32 bytes (256 bits) is a common recommendation.
    if num_bytes < 32:
        raise ValueError(f"Input 'num_bytes' must be at least 32 for adequate security, got {num_bytes}.")

    # --- Generate and Return Key ---
    # secrets.token_hex generates num_bytes random bytes and returns them
    # as a hexadecimal string (length will be 2 * num_bytes).
    return secrets.token_hex(num_bytes)

# --- Example Usage (when running the script directly) ---
if __name__ == "__main__":
    try:
        # --- Generate Standard Key ---
        key_length_bytes = 32
        secret_key = generate_secure_hex_key(key_length_bytes)

        print("\n--- Generated Flask Secret Key ---")
        print("Copy the following line and set it as your FLASK_SECRET_KEY")
        print("in your .env file or configuration:")
        # Enclose in quotes for robustness in .env files
        print(f'\nFLASK_SECRET_KEY="{secret_key}"\n')
        print(f"(Generated from {key_length_bytes} random bytes, resulting in {len(secret_key)} hex characters)")
        print("--- Keep this key secret and secure! ---")

        # --- Generate Longer Key Example ---
        # key_length_bytes_long = 48
        # secret_key_long = generate_secure_hex_key(key_length_bytes_long)
        # print(f"\nExample longer key ({key_length_bytes_long} bytes -> {len(secret_key_long)} chars):")
        # print(f'FLASK_SECRET_KEY="{secret_key_long}"')

    except (TypeError, ValueError) as e:
        print(f"\nError generating key: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}", file=sys.stderr)
        sys.exit(1)

    # --- Example Error Handling Tests (Optional) ---
    # print("\n--- Testing Error Cases ---")
    # try:
    #     generate_secure_hex_key("abc")
    # except TypeError as e:
    #     print(f"Caught expected TypeError: {e}")
    # try:
    #     generate_secure_hex_key(16) # Too short
    # except ValueError as e:
    #     print(f"Caught expected ValueError: {e}")

