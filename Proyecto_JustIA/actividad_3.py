from actividad_2 import clasificar_consulta

def mostrar_interfaz():
    print("\n" + "="*40)
    print("   BIENVENIDO AL SISTEMA JUSTIA")
    print("="*40)
    
    while True:
        print("\n1. Clasificar caso")
        print("2. Cargar documento (Simulado)")
        print("3. Volver al menú principal")
        
        opc = input("\nSeleccione una opción: ")
        
        if opc == "1":
            texto = input("Describa el problema legal: ")
            categoria = clasificar_consulta(texto)
            print(f"\n[Resultado IA]: El área probable es: {categoria}")
            print("[Info]: Basado en normativa colombiana vigente.")
        elif opc == "2":
            ruta = input("Nombre del archivo (ej: caso1.pdf): ")
            print(f"Leyendo {ruta}... Extrayendo información.")
        elif opc == "3":
            break
        else:
            print("Opción inválida.")