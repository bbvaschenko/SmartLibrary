import base64

# Ваш текущий ключ из .env
encoded_key = "MDE5YjQ3OWUtZDE1Zi03M2EwLWI3OWItODE2MDg1MDM5N2VhOmY5MzJiNTUwLWVkMjEtNDM5MS1hOWRiLTFiYjIyZTlkOTlmYg=="

try:
    # Декодируем Base64
    decoded_bytes = base64.b64decode(encoded_key)
    decoded_str = decoded_bytes.decode('utf-8')

    print(f"🔑 Декодированная строка:")
    print(f"'{decoded_str}'")
    print(f"\nДлина: {len(decoded_str)} символов")

    # Пробуем разделить по двоеточию
    if ":" in decoded_str:
        parts = decoded_str.split(":", 1)
        print(f"\n✅ Найдено 2 части:")
        print(f"Client ID:  {parts[0]}")
        print(f"Client Secret: {parts[1]}")
    else:
        print("\n⚠️  Не найден разделитель ':'")
        print("Возможно, это единый ключ другого формата")

except Exception as e:
    print(f"❌ Ошибка декодирования: {e}")