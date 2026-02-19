import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(
    page_title="ВРУ IX — Аналіз доброчесності",
    page_icon="🇺🇦",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
.section-divider { border-top: 2px solid #3d4266; margin: 24px 0; }
</style>""", unsafe_allow_html=True)

SCORE_LABELS = {
    0: "🟢 Чистий",
    1: "🟡 Мінорні (медіа розслідування)",
    2: "🔴 Підозри НАБУ / САП / НАЗК",
    3: "⛔ Вирок ВАКС"
}
SCORE_COLORS = ["#2ecc71", "#f39c12", "#e74c3c", "#8e44ad"]

@st.cache_data
def load_and_prepare(path: str = "data.csv") -> pd.DataFrame:
    df = pd.read_csv(path)

    df["Birth_Year"] = pd.to_numeric(df["Birth_Year"], errors="coerce")
    df["Birth_Year"] = df["Birth_Year"].fillna(df["Birth_Year"].median())
    df["Age_at_Election"] = 2019 - df["Birth_Year"]
    df["Age_Group"] = pd.cut(df["Age_at_Election"],
                              bins=[0,35,45,55,65,100],
                              labels=["<35","35–44","45–54","55–64","65+"])

    for c in ["System","Business","Civil Society","Media","Military"]:
        df[f"is_{c}"] = df["Career_Origin"].astype(str).apply(lambda x: 1 if c in x else 0)
    df["Career_Multi"] = (df["Career_Origin"].str.count(",") >= 1).astype(int)

    HIGH = {"Finance","Energy","Agrarian","Budget","State Power"}
    MID  = {"Law Enforcement","Transport","Legal Policy","Anti-Corruption"}
    df["Committee_Risk"] = df["Committee_Short"].apply(
        lambda x: "🔴 High" if x in HIGH else ("🟡 Medium" if x in MID else "🟢 Low"))
    df["Committee_Risk_Num"] = df["Committee_Short"].apply(
        lambda x: 2 if x in HIGH else (1 if x in MID else 0))

    df["Integrity_Score"] = pd.to_numeric(df["Integrity_Score"], errors="coerce").fillna(0).astype(int)
    df["is_Corrupt"]      = (df["Integrity_Score"] >= 2).astype(int)
    df["Integrity_Label"] = df["Integrity_Score"].map(SCORE_LABELS)

    pm = {"Sluha Narodu":"Слуга народу","Sluga Narodu":"Слуга народу",
          "OPZZH":"ОПЗЖ","Opposition Block":"Опозиційний блок"}
    df["Party"] = df["Fraction_2019"].replace(pm).fillna("Самовисуванці")

    def macro(x):
        x = str(x)
        if any(r in x for r in ["Lviv","Ternopil","Volyn","Ivano","Chernivtsi","Zakarpattia"]): return "Захід"
        if any(r in x for r in ["Donetsk","Luhansk","Kharkiv","Dnipro","Zaporizhzhia"]): return "Схід"
        if any(r in x for r in ["Odesa","Mykolaiv","Kherson","Crimea"]): return "Південь"
        if any(r in x for r in ["Kyiv","Cherkasy","Zhytomyr","Vinnytsia","Kirovohrad","Poltava"]): return "Центр"
        if any(r in x for r in ["Chernihiv","Sumy"]): return "Північ"
        return "Інше"
    df["Region"] = df["Region_Origin"].apply(macro)

    if "Status" in df.columns:
        df["Early_Termination"] = (df["Status"] == "Terminated").astype(int)
    else:
        df["Early_Termination"] = 0

    return df

# ── SIDEBAR ───────────────────────────────────────────────────
st.sidebar.title("🇺🇦 ВРУ IX Аналіз")
st.sidebar.markdown("---")
st.sidebar.markdown("**Integrity Score:**")
for k, v in SCORE_LABELS.items():
    st.sidebar.markdown(f"{v}")
st.sidebar.markdown("---")

data_path = st.sidebar.text_input("Шлях до data.csv", value="final_9skl_data.csv")
score_threshold = st.sidebar.radio(
    "Поріг 'корупціонер'", options=[2, 1],
    format_func=lambda x: f"Score ≥ {x}", index=0)

try:
    df = load_and_prepare(data_path)

    all_parties = sorted(df["Party"].unique().tolist())
    selected_parties = st.sidebar.multiselect("Фільтр партій", all_parties, default=all_parties)
    df_f = df[df["Party"].isin(selected_parties)].copy() if selected_parties else df.copy()
    df_f["is_Corrupt"] = (df_f["Integrity_Score"] >= score_threshold).astype(int)

    # ══════════════════════════════════════════════════════════
    # HEADER
    # ══════════════════════════════════════════════════════════
    st.title("🇺🇦 ВРУ IX скликання — Predictive Integrity Dashboard")
    st.caption("Ordinal Logistic Regression | Ex-ante Model | Для Тимофія Милованова")
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════
    # SECTION 1 — KPI
    # ══════════════════════════════════════════════════════════
    st.header("📊 Загальна статистика")

    n     = len(df_f)
    n0    = (df_f["Integrity_Score"] == 0).sum()
    n1    = (df_f["Integrity_Score"] == 1).sum()
    n2    = (df_f["Integrity_Score"] == 2).sum()
    n3    = (df_f["Integrity_Score"] == 3).sum()
    n_cor = df_f["is_Corrupt"].sum()

    c1,c2,c3,c4,c5,c6 = st.columns(6)
    c1.metric("👤 Всього", n)
    c2.metric("🟢 Чисті (Score=0)", n0, f"{n0/n*100:.1f}%")
    c3.metric("🟡 Медіа розслідування (Score=1)", n1, f"{n1/n*100:.1f}%")
    c4.metric("🔴 НАБУ/САП/НАЗК (Score=2)", n2, f"{n2/n*100:.1f}%")
    c5.metric("⛔ Вирок ВАКС (Score=3)", n3, f"{n3/n*100:.1f}%")
    c6.metric(f"🚨 Корупціонери (≥{score_threshold})", n_cor, f"{n_cor/n*100:.1f}%")

    col1, col2 = st.columns([3,2])
    with col1:
        sc = df_f["Integrity_Score"].value_counts().sort_index().reset_index()
        sc.columns = ["Score","Count"]
        sc["Label"] = sc["Score"].map(SCORE_LABELS)
        fig = px.bar(sc, x="Label", y="Count", color="Score",
                     color_continuous_scale=SCORE_COLORS,
                     text="Count", title="Розподіл Integrity Score")
        fig.update_traces(textposition="outside")
        fig.update_layout(showlegend=False, xaxis_title="", coloraxis_showscale=False)
        st.plotly_chart(fig, width="stretch")
    with col2:
        fig2 = px.pie(sc, values="Count", names="Label",
                      color_discrete_sequence=SCORE_COLORS, title="Частки по Score")
        st.plotly_chart(fig2, width="stretch")

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════
    # SECTION 2 — BY PARTY
    # ══════════════════════════════════════════════════════════
    st.header("🏛️ Розподіл по партіям")

    ps = df_f.groupby("Party", as_index=False).agg(
        total     =("Integrity_Score","count"),
        corrupt   =("is_Corrupt","sum"),
        mean_score=("Integrity_Score","mean"),
        n0=("Integrity_Score", lambda x:(x==0).sum()),
        n1=("Integrity_Score", lambda x:(x==1).sum()),
        n2=("Integrity_Score", lambda x:(x==2).sum()),
        n3=("Integrity_Score", lambda x:(x==3).sum()),
    )
    ps["corrupt_pct"] = (ps["corrupt"]/ps["total"]*100).round(1)
    ps = ps.sort_values("corrupt_pct", ascending=False)

    col1, col2 = st.columns(2)
    with col1:
        fig = px.bar(ps, x="Party", y="corrupt_pct",
                     color="corrupt_pct", color_continuous_scale="RdYlGn_r",
                     text="corrupt_pct",
                     title=f"% корупціонерів (Score≥{score_threshold}) по партіям",
                     labels={"corrupt_pct":"% корупціонерів","Party":"Партія"})
        fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
        fig.update_layout(xaxis_tickangle=30, coloraxis_showscale=False)
        st.plotly_chart(fig, width="stretch")
    with col2:
        melt = ps[["Party","n0","n1","n2","n3"]].melt(id_vars="Party",var_name="Score",value_name="Count")
        melt["Score"] = melt["Score"].map({"n0":SCORE_LABELS[0],"n1":SCORE_LABELS[1],
                                            "n2":SCORE_LABELS[2],"n3":SCORE_LABELS[3]})
        fig = px.bar(melt, x="Party", y="Count", color="Score",
                     color_discrete_sequence=SCORE_COLORS,
                     title="Stacked розподіл Score по партіям",
                     labels={"Count":"Кількість","Party":"Партія"})
        fig.update_layout(xaxis_tickangle=30)
        st.plotly_chart(fig, width="stretch")

    st.dataframe(
        ps.rename(columns={"Party":"Партія","total":"Всього","n0":"Score=0","n1":"Score=1",
                             "n2":"Score=2","n3":"Score=3","corrupt":"Корупція≥2",
                             "corrupt_pct":"% Корупція","mean_score":"Сер. Score"})
          .reset_index(drop=True),
        width="stretch", hide_index=True)

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════
    # SECTION 3 — BY COMMITTEE
    # ══════════════════════════════════════════════════════════
    st.header("🏢 Розподіл по комітетам")

    cs = df_f.groupby(["Committee_Short","Committee_Risk","Committee_Risk_Num"], as_index=False).agg(
        total     =("Integrity_Score","count"),
        corrupt   =("is_Corrupt","sum"),
        mean_score=("Integrity_Score","mean")
    )
    cs["corrupt_pct"] = (cs["corrupt"]/cs["total"]*100).round(1)
    cs = cs.sort_values("corrupt_pct", ascending=False)

    fig = px.bar(cs, x="Committee_Short", y="corrupt_pct",
             color="Committee_Risk",
             color_discrete_map={"🔴 High":"#e74c3c","🟡 Medium":"#f39c12","🟢 Low":"#2ecc71"},
             text="corrupt_pct",
             title="% корупціонерів по комітетам (🔴High / 🟡Medium / 🟢Low risk tier)",
             labels={"corrupt_pct":"% корупціонерів","Committee_Short":"Комітет","Committee_Risk":"Ризик"},
             category_orders={"Committee_Short": cs["Committee_Short"].tolist()})
    fig.update_traces(texttemplate="%{text:.0f}%", textposition="outside")
    fig.update_layout(xaxis_tickangle=45)
    st.plotly_chart(fig, width="stretch")

    st.subheader("🗺️ Heatmap: Комітет × Партія")
    pivot = df_f.pivot_table(values="is_Corrupt", index="Committee_Short",
                              columns="Party", aggfunc="mean", fill_value=0)*100
    fig = px.imshow(pivot, color_continuous_scale="RdYlGn_r", zmin=0, zmax=100,
                    text_auto=".0f", title="% корупціонерів: Комітет × Партія",
                    labels={"color":"% корупціонерів"})
    fig.update_layout(height=500)
    st.plotly_chart(fig, width="stretch")

    sel = st.selectbox("Деталізація по комітету:", ["Всі"] + sorted(df_f["Committee_Short"].unique().tolist()))
    if sel != "Всі":
        cp = df_f[df_f["Committee_Short"]==sel].groupby("Party", as_index=False).agg(
            corrupt=("is_Corrupt","sum"), total=("Integrity_Score","count"))
        cp["corrupt_pct"] = (cp["corrupt"]/cp["total"]*100).round(1)
        fig = px.bar(cp.sort_values("corrupt_pct", ascending=False),
                     x="Party", y="corrupt_pct", color="Party", text="corrupt_pct",
                     title=f"Комітет '{sel}': % корупціонерів по партіям")
        fig.update_traces(texttemplate="%{text:.0f}%", textposition="outside")
        st.plotly_chart(fig, width="stretch")

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════
    # SECTION 4 — OTHER VARIABLES
    # ══════════════════════════════════════════════════════════
    st.header("📈 Інші змінні")

    def simple_bar(groupby_col, label_map, title, color_seq=None):
        g = df_f.groupby(groupby_col)["is_Corrupt"].mean().reset_index()
        g.columns = [groupby_col, "pct"]
        g["Підпис"] = g[groupby_col].map(label_map) if label_map else g[groupby_col].astype(str)
        g["pct_fmt"] = (g["pct"]*100).round(1)
        kwargs = dict(color_discrete_sequence=color_seq) if color_seq else dict(color="pct_fmt", color_continuous_scale="RdYlGn_r")
        fig = px.bar(g.sort_values("pct_fmt", ascending=False), x="Підпис", y="pct_fmt",
                     text="pct_fmt", title=title,
                     labels={"pct_fmt":"% корупціонерів","Підпис":""}, **kwargs)
        fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
        if not color_seq:
            fig.update_layout(coloraxis_showscale=False)
        return fig

    col1, col2, col3 = st.columns(3)
    with col1:
        st.plotly_chart(simple_bar("Gender_Male",{1:"Чоловіки 👨",0:"Жінки 👩"},
                        "% корупціонерів за статтю", ["#3498db","#e91e8c"]), width="stretch")
    with col2:
        st.plotly_chart(simple_bar("Incumbency",{0:"Новачок 🆕",1:"Повторно 🔄"},
                        "Повторно обрані vs нові"), width="stretch")
    with col3:
        st.plotly_chart(simple_bar("Party_Switcher",{0:"Стабільний 🏠",1:"Перебіжчик ⚡"},
                        "Party Switcher vs Стабільні", ["#2ecc71","#e74c3c"]), width="stretch")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.plotly_chart(simple_bar("Region", None, "% корупціонерів по регіонах"), width="stretch")
    with col2:
        career_rows = []
        for c_col, c_label in [("is_Business","Бізнес"),("is_System","Система"),
                                ("is_Media","Медіа"),("is_Civil Society","Громадянське"),
                                ("is_Military","Військові"),("Career_Multi","Multi-sector")]:
            if c_col in df_f.columns:
                pct = df_f[df_f[c_col]==1]["is_Corrupt"].mean()
                career_rows.append({"Тип": c_label, "pct": 0 if np.isnan(pct) else pct*100})
        c_df = pd.DataFrame(career_rows).sort_values("pct", ascending=False)
        fig = px.bar(c_df, x="Тип", y="pct", text="pct",
                     color="pct", color_continuous_scale="RdYlGn_r",
                     title="% корупціонерів за типом кар'єри",
                     labels={"pct":"% корупціонерів","Тип":""})
        fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
        fig.update_layout(coloraxis_showscale=False)
        st.plotly_chart(fig, width="stretch")
    with col3:
        ag = df_f.groupby("Age_Group", as_index=False)["is_Corrupt"].mean().dropna()
        ag["pct"] = (ag["is_Corrupt"]*100).round(1)
        fig = px.bar(ag, x="Age_Group", y="pct", text="pct",
                     color="pct", color_continuous_scale="RdYlGn_r",
                     title="% корупціонерів за віковою групою",
                     labels={"pct":"% корупціонерів","Age_Group":"Вік"})
        fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
        fig.update_layout(coloraxis_showscale=False)
        st.plotly_chart(fig, width="stretch")

    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(simple_bar("List_Member",{0:"Мажоритарник 🗳️",1:"Списочник 📋"},
                        "Списочники vs мажоритарники"), width="stretch")
    with col2:
        # Education_Dummy = профільна освіта до депутатства (0=непрофільна, 1=профільна)
        st.plotly_chart(simple_bar("Education_Dummy",
                        {0:"Непрофільна освіта", 1:"Профільна до депутатства 🎓"},
                        "Профільна vs непрофільна освіта до депутатства"), width="stretch")

    fig_v = px.violin(df_f, y="Age_at_Election", x="Integrity_Label",
                      color="Integrity_Label",
                      color_discrete_sequence=SCORE_COLORS,
                      box=True, points="all",
                      title="Розподіл віку депутатів за Integrity Score",
                      labels={"Age_at_Election":"Вік (2019)","Integrity_Label":""})
    st.plotly_chart(fig_v, width="stretch")

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════
    # SECTION 5 — MODEL STATS
    # ══════════════════════════════════════════════════════════
    st.header("🤖 Ex-ante Predictive Model")
    st.info("Committee data залишено (52% R²) | Valid для Дня 1 скликання")

    model_df = pd.DataFrame({
        "Модель": ["Оригінальна baseline","Full post-hoc","Ex-ante (base)","Ex-ante + interactions 🏆"],
        "Pseudo R²": [0.030, 0.258, 0.246, 0.249],
        "Features": [12, 45, 40, 43],
        "Endogenous?": ["❌ Так","❌ Так","✅ Ні","✅ Ні"],
        "Застосування": ["Ретроспектива","Повний аналіз","Прогноз День 1","Прогноз enhanced 🏆"],
        "vs baseline": ["–","+759%","+720%","+730%"]
    })
    st.dataframe(model_df, width="stretch", hide_index=True)

    fig = go.Figure(go.Bar(
        x=model_df["Модель"], y=model_df["Pseudo R²"],
        marker_color=["#e74c3c","#f39c12","#3498db","#2ecc71"],
        text=model_df["Pseudo R²"], texttemplate="%{text:.3f}", textposition="outside"))
    fig.update_layout(title="Pseudo R² по версіям моделі", yaxis_range=[0,0.32], yaxis_title="Pseudo R²")
    st.plotly_chart(fig, width="stretch")

    col1, col2 = st.columns(2)
    with col1:
        r2c = pd.DataFrame({
            "Компонент":["Committee_Risk","Committee dummies","Party/Fraction",
                          "Demographics","Career type","Region","Interactions"],
            "R²":[0.08,0.05,0.04,0.03,0.03,0.02,0.003],
            "Категорія":["Committee","Committee","Party","Demographics","Career","Region","Interaction"]
        })
        fig = px.pie(r2c, values="R²", names="Компонент", color="Категорія",
                     color_discrete_map={"Committee":"#e74c3c","Party":"#3498db",
                                          "Demographics":"#2ecc71","Career":"#f39c12",
                                          "Region":"#9b59b6","Interaction":"#1abc9c"},
                     title="Внесок категорій в R²=0.249  (Committee = 52% ✅)")
        st.plotly_chart(fig, width="stretch")

    with col2:
        ft = pd.DataFrame({
            "Предиктор":["Party_Switcher","is_Business","Region_North","Committee_Risk",
                          "Comm_Finance","Comm_Eco","Frac_Self-nominated",
                          "is_Business×Comm_Finance (neg)","is_System×Comm_Risk (neg)",
                          "Party_Sw.×Frac_Self-nom. (neg)"],
            "OR":[18.2,5.01,4.61,4.49,5.25,5.07,4.01,0.08,0.41,0.037],
            "Тип":["Party","Career","Region","Committee","Committee","Committee",
                    "Party","Interaction","Interaction","Interaction"]
        }).sort_values("OR")
        fig = px.bar(ft, y="Предиктор", x="OR", orientation="h", color="Тип",
                     color_discrete_map={"Party":"#3498db","Committee":"#e74c3c",
                                          "Career":"#f39c12","Region":"#9b59b6","Interaction":"#1abc9c"},
                     title="TOP предикторів — Odds Ratio")
        fig.add_vline(x=1, line_dash="dash", line_color="gray", annotation_text="OR=1")
        st.plotly_chart(fig, width="stretch")

    st.subheader("📋 Bootstrap 95% CI — значущі предиктори")
    boot_df = pd.DataFrame({
        "Предиктор":["Party_Switcher","Frac_Holos","Frac_Opposition Bloc","Committee_Risk",
                      "is_System×Committee_Risk","is_Business×Comm_Finance",
                      "Party_Sw.×Frac_Self-nom.","Comm_Eco","Comm_Finance",
                      "Region_North","Frac_Self-nominated","Age_at_Election","Age_Squared"],
        "β":[2.960,-3.816,-3.945,1.502,-0.897,-2.515,-3.273,1.623,1.658,1.527,1.389,0.279,-0.003],
        "CI Lower":[2.010,-9.059,-8.696,0.811,-1.539,-5.030,-5.902,0.220,0.252,0.139,0.154,0.011,-0.006],
        "CI Upper":[4.079,-0.134,-0.010,2.257,-0.349,-0.542,-0.350,3.490,3.382,3.220,2.915,0.539,0.000],
    })
    fig = go.Figure()
    for _, row in boot_df.iterrows():
        c = "#2ecc71" if row["β"] > 0 else "#e74c3c"
        fig.add_trace(go.Scatter(x=[row["CI Lower"],row["CI Upper"]],y=[row["Предиктор"]]*2,
                                  mode="lines",line=dict(color=c,width=5),showlegend=False))
        fig.add_trace(go.Scatter(x=[row["β"]],y=[row["Предиктор"]],
                                  mode="markers",marker=dict(color=c,size=10),showlegend=False))
    fig.add_vline(x=0,line_dash="dash",line_color="white",opacity=0.4)
    fig.update_layout(title="Bootstrap 95% CI",xaxis_title="Coefficient β",height=420)
    st.plotly_chart(fig, width="stretch")

    diag, notes = st.columns(2)
    with diag:
        st.dataframe(pd.DataFrame({
            "Метрика":["Pseudo R² (McFadden)","CV Accuracy (Repeated 5-Fold)","ROC-AUC","Permutation p","Hosmer-Lemeshow p"],
            "Значення":["0.249","80.6% ± 2.4%","≥0.82","<0.001","0.09+"],
            "Статус":["✅ Strong","✅ Excellent","✅ Good","✅ Significant","✅ Good fit"]
        }), width="stretch", hide_index=True)
    with notes:
        st.markdown("""
**Легенда Integrity Score:**
| Score | Значення |
|-------|----------|
| 0 | 🟢 Чистий — жодних даних |
| 1 | 🟡 Мінорні — розслідування авторитетних медіа |
| 2 | 🔴 Корупція — підозри НАБУ / САП / НАЗК |
| 3 | ⛔ Тяжка — вирок ВАКС або інших судів |

**Committee = 52% R²** — включення виправдане (призначення на тиждень 1–2)  
**Party_Switcher** — найсильніший предиктор (OR = 18.2)
""")

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════
    # SECTION 6 — RISK CALCULATOR
    # ══════════════════════════════════════════════════════════
    st.header("🔮 Risk Calculator — Day 1 Prediction")

    r1, r2, r3 = st.columns(3)
    with r1:
        r_age      = st.slider("Вік", 25, 80, 45)
        r_switcher = st.checkbox("⚡ Перебіжчик між партіями")
        r_business = st.checkbox("💼 Бізнес-бекграунд")
        r_system   = st.checkbox("🏛️ Системний (держслужба)")
    with r2:
        r_comm = st.selectbox("🏢 Ризик комітету",
            ["🟢 Low (Social/Foreign/Defense)","🟡 Medium (Law/Transport)","🔴 High (Finance/Energy/Budget)"])
        r_region = st.selectbox("📍 Регіон", ["Центр","Захід","Схід","Південь","Північ","Інше"])
        r_inc    = st.checkbox("🔄 Повторно обраний")
    with r3:
        r_list   = st.radio("Тип обрання",["Списочник","Мажоритарник"], horizontal=True)
        r_occ    = st.checkbox("🚩 З окупованих територій")
        r_gender = st.radio("Стать",["Чоловік","Жінка"], horizontal=True)

    if st.button("🔍 Оцінити ризик", type="primary"):
        score, reasons = 0.0, []
        if r_switcher: score += 2.96; reasons.append("🚨 Party Switcher β=+2.96 (OR=18.2)")
        if r_business: score += 1.61; reasons.append("⚠️ Бізнес β=+1.61 (OR=5.0)")
        if "High" in r_comm: score += 1.50; reasons.append("⚠️ High-risk комітет β=+1.50")
        elif "Medium" in r_comm: score += 0.50
        if r_region == "Північ": score += 1.53; reasons.append("⚠️ Північ β=+1.53 (OR=4.6)")
        if r_system: score += 1.23; reasons.append("ℹ️ Системний β=+1.23")
        if r_inc: score -= 1.0; reasons.append("✅ Повторно обраний (стабілізуючий)")
        if r_occ: score += 0.62; reasons.append("ℹ️ З окупованих β=+0.62")
        if r_gender == "Чоловік": score += 0.50

        p = 1 / (1 + np.exp(-score))
        if p > 0.6:
            st.error(f"**🔴 ВИСОКИЙ РИЗИК** | P(Score≥2) ≈ {p:.1%}")
            st.error("🚨 ACTION: Top-50 → НАБУ + медіа моніторинг")
        elif p > 0.35:
            st.warning(f"**🟡 СЕРЕДНІЙ РИЗИК** | P(Score≥2) ≈ {p:.1%}")
            st.warning("📊 ACTION: Стандартний моніторинг + квартальний review")
        else:
            st.success(f"**🟢 НИЗЬКИЙ РИЗИК** | P(Score≥2) ≈ {p:.1%}")
            st.success("✅ ACTION: Базовий моніторинг")

        if reasons:
            st.markdown("**Фактори ризику:**")
            for r in reasons: st.markdown(f"  - {r}")

except FileNotFoundError:
    st.error(f"❌ Файл не знайдено: `{data_path}`")
    st.info("Вкажіть правильний шлях у sidebar")
