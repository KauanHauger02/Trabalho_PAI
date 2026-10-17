import customtkinter as ctk
import nibabel as nib
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure


class VisualizadorNii:
    """Carrega volumes NIfTI e controla sua visualização dentro de um frame."""

    def __init__(self, frame_pai):
        self.frame_pai = frame_pai

        self.dados_nii = None
        self.dados_segmentação = None
        self.fatia_atual = 0
        self.eixo_atual = "Axial"

        self.figura = None
        self.eixo_mri = None
        self.imagem_plot = None
        self.canvas_plot = None
        self.slider_fatia = None
        self.label_fatia = None

        self.mostrar_mensagem_inicial()

    def limpar_frame(self):
        """Remove os widgets atuais da área de visualização."""
        for widget in self.frame_pai.winfo_children():
            widget.destroy()

    def mostrar_mensagem_inicial(self):
        self.limpar_frame()
        mensagem = ctk.CTkLabel(
            self.frame_pai,
            text="Selecione um arquivo .nii para visualizar",
            font=("Arial", 16),
        )
        mensagem.pack(expand=True)

    @staticmethod
    def obter_fatia_mri(dados, eixo, numero_fatia):
        """Retorna uma fatia 2D do volume 3D conforme o plano selecionado."""
        if eixo == "Axial":
            return dados[:, :, numero_fatia]
        if eixo == "Coronal":
            return dados[:, numero_fatia, :]
        if eixo == "Sagital":
            return dados[numero_fatia, :, :]
        raise ValueError(f"Eixo de visualização desconhecido: {eixo}")

    @staticmethod
    def obter_total_fatias(dados, eixo):
        """Retorna a quantidade de fatias disponível no eixo escolhido."""
        if eixo == "Axial":
            return dados.shape[2]
        if eixo == "Coronal":
            return dados.shape[1]
        if eixo == "Sagital":
            return dados.shape[0]
        raise ValueError(f"Eixo de visualização desconhecido: {eixo}")

    def carregar_arquivo(self, caminho_arquivo):
        """Carrega um volume .nii e monta os controles de visualização."""
        try:
            imagem = nib.load(caminho_arquivo)
            dados = imagem.get_fdata()

            if dados.ndim != 3:
                raise ValueError(
                    f"Era esperado um volume 3D, mas o arquivo possui "
                    f"{dados.ndim} dimensões."
                )

            self.dados_nii = dados
            self.eixo_atual = "Axial"
            self.fatia_atual = self.obter_total_fatias(
                self.dados_nii, self.eixo_atual
            ) // 2

            self.montar_visualizacao()

        except Exception as erro:
            self.dados_nii = None
            self.limpar_frame()
            ctk.CTkLabel(
                self.frame_pai,
                text=f"Erro ao abrir NIfTI:\n{erro}",
                text_color="red",
                wraplength=600,
            ).pack(expand=True, padx=20, pady=20)

    def montar_visualizacao(self):
        """Cria o gráfico Matplotlib e os controles dentro do frame pai."""
        self.limpar_frame()

        cor_fundo = (
            "#2b2b2b"
            if ctk.get_appearance_mode() == "Dark"
            else "#f0f0f0"
        )

        self.figura = Figure(figsize=(5, 5), facecolor=cor_fundo)
        self.eixo_mri = self.figura.add_subplot(111)
        self.eixo_mri.axis("off")

        fatia = self.obter_fatia_mri(
            self.dados_nii, self.eixo_atual, self.fatia_atual
        )
        self.imagem_plot = self.eixo_mri.imshow(fatia, cmap="gray")
        self.figura.tight_layout()

        self.canvas_plot = FigureCanvasTkAgg(
            self.figura, master=self.frame_pai
        )
        self.canvas_plot.get_tk_widget().pack(
            fill="both", expand=True, padx=10, pady=10
        )

        frame_controles = ctk.CTkFrame(
            self.frame_pai, fg_color="transparent"
        )
        frame_controles.pack(fill="x", padx=30, pady=5)

        ctk.CTkLabel(
            frame_controles,
            text="Eixo de visualização:",
            font=("Arial", 12, "bold"),
        ).pack(side="left", padx=5)

        self.menu_eixo = ctk.CTkOptionMenu(
            frame_controles,
            values=["Axial", "Coronal", "Sagital"],
            command=self.alterar_eixo,
        )
        self.menu_eixo.set(self.eixo_atual)
        self.menu_eixo.pack(side="left", padx=5)

        total_fatias = self.obter_total_fatias(
            self.dados_nii, self.eixo_atual
        )
        self.label_fatia = ctk.CTkLabel(
            self.frame_pai,
            text=f"Fatia: {self.fatia_atual} / {total_fatias - 1}",
            font=("Arial", 14),
        )
        self.label_fatia.pack(pady=5)

        self.slider_fatia = ctk.CTkSlider(
            self.frame_pai,
            from_=0,
            to=max(total_fatias - 1, 1),
            number_of_steps=max(total_fatias - 1, 1),
            command=self.atualizar_fatia,
        )
        self.slider_fatia.set(self.fatia_atual)
        self.slider_fatia.pack(fill="x", padx=30, pady=10)

        self.canvas_plot.draw()

    def atualizar_fatia(self, valor):
        """Atualiza a imagem quando o usuário move o controle deslizante."""
        if self.dados_nii is None:
            return

        self.fatia_atual = int(float(valor))
        total_fatias = self.obter_total_fatias(
            self.dados_nii, self.eixo_atual
        )
        nova_fatia = self.obter_fatia_mri(
            self.dados_nii, self.eixo_atual, self.fatia_atual
        )

        self.imagem_plot.set_data(nova_fatia)
        self.imagem_plot.set_extent(
            [0, nova_fatia.shape[1], nova_fatia.shape[0], 0]
        )
        self.label_fatia.configure(
            text=f"Fatia: {self.fatia_atual} / {total_fatias - 1}"
        )
        self.canvas_plot.draw_idle()

    def alterar_eixo(self, novo_eixo):
        """Troca o plano anatômico e reposiciona a fatia no centro."""
        if self.dados_nii is None:
            return

        self.eixo_atual = novo_eixo
        total_fatias = self.obter_total_fatias(
            self.dados_nii, self.eixo_atual
        )
        self.fatia_atual = total_fatias // 2

        self.slider_fatia.configure(
            from_=0,
            to=max(total_fatias - 1, 1),
            number_of_steps=max(total_fatias - 1, 1),
        )
        self.slider_fatia.set(self.fatia_atual)

        nova_fatia = self.obter_fatia_mri(
            self.dados_nii, self.eixo_atual, self.fatia_atual
        )
        self.imagem_plot.set_data(nova_fatia)
        self.imagem_plot.set_extent(
            [0, nova_fatia.shape[1], nova_fatia.shape[0], 0]
        )

        self.label_fatia.configure(
            text=f"Fatia: {self.fatia_atual} / {total_fatias - 1}"
        )
        self.canvas_plot.draw_idle()
