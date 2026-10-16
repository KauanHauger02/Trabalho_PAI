import os
import customtkinter as ctk
from tkinter import filedialog

from visualizador import VisualizadorNii


TARGET_EXTENSIONS = (".nii", ".nii.gz")

ctk.set_appearance_mode("System")
ctk.set_default_color_theme("green")


class InterfaceMRI:
    """Define a janela e os controles principais da aplicação."""

    def __init__(self, root):
        self.root = root
        self.root.geometry("1024x650")
        self.root.title("Visualizador de Imagens NIfTI")

        self.criar_layout()

    def criar_layout(self):
        self.frame_esquerdo = ctk.CTkFrame(self.root, width=300)
        self.frame_esquerdo.pack(
            side="left", fill="y", padx=15, pady=10
        )
        self.frame_esquerdo.pack_propagate(False)

        self.botao_pasta = ctk.CTkButton(
            self.frame_esquerdo,
            text="Selecionar pasta MRI",
            command=self.selecionar_pasta,
        )
        self.botao_pasta.pack(
            pady=20, padx=10, fill="x"
        )

        self.lista_arquivos = ctk.CTkScrollableFrame(
            self.frame_esquerdo,
            label_text="Arquivos disponíveis",
        )
        self.lista_arquivos.pack(
            pady=10, padx=10, fill="both", expand=True
        )

        self.frame_visualizador = ctk.CTkFrame(self.root)
        self.frame_visualizador.pack(
            side="right", fill="both", expand=True, padx=15, pady=10
        )

        # A interface delega leitura de NIfTI e desenho ao outro módulo.
        self.visualizador = VisualizadorNii(self.frame_visualizador)

    def selecionar_pasta(self):
        pasta = filedialog.askdirectory(title="Selecione a pasta das imagens")

        if not pasta:
            return

        for widget in self.lista_arquivos.winfo_children():
            widget.destroy()

        try:
            arquivos = sorted(
                nome
                for nome in os.listdir(pasta)
                if os.path.isfile(os.path.join(pasta, nome))
                and nome.lower().endswith(TARGET_EXTENSIONS)
            )

            if not arquivos:
                ctk.CTkLabel(
                    self.lista_arquivos,
                    text="Nenhum arquivo .nii ou .nii.gz encontrado.",
                    wraplength=220,
                ).pack(pady=20, padx=10)
                return

            for nome_arquivo in arquivos:
                caminho = os.path.join(pasta, nome_arquivo)
                ctk.CTkButton(
                    self.lista_arquivos,
                    text=nome_arquivo,
                    anchor="w",
                    fg_color="transparent",
                    text_color=("black", "white"),
                    hover_color=("gray85", "gray25"),
                    command=lambda caminho=caminho: (
                        self.visualizador.carregar_arquivo(caminho)
                    ),
                ).pack(fill="x", padx=10, pady=2)

        except OSError as erro:
            ctk.CTkLabel(
                self.lista_arquivos,
                text=f"Erro ao ler a pasta:\n{erro}",
                text_color="red",
                wraplength=220,
            ).pack(pady=20, padx=10)


def main():
    root = ctk.CTk()
    InterfaceMRI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
