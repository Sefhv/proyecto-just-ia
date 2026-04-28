import actividad_1
import actividad_2
import actividad_3

def main():
    while True:
        print("\n--- PANEL DE CONTROL JUSTIA ---")
        print("1. Ejecutar Fase 1 (Limpieza de Datos)")
        print("2. Ejecutar Fase 2 (Configurar Diccionario)")
        print("3. Abrir Interfaz de Usuario (Fase 3)")
        print("4. Salir")
        
        opcion = input("\nSeleccione fase a ejecutar: ")
        
        if opcion == "1":
            actividad_1.ejecutar_actividad_1()
        elif opcion == "2":
            actividad_2.crear_diccionario_base()
        elif opcion == "3":
            actividad_3.mostrar_interfaz()
        elif opcion == "4":
            print("Saliendo del sistema...")
            break
        else:
            print("Opción no válida.")

if __name__ == "__main__":
    main()