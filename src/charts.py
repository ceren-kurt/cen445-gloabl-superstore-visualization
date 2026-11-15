import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.data_loader import (
    CAT,
    DISCOUNT_LABELS,
    NUM,
    margin_by,
    segment_customers,
    train_model,
)

CAT_COLORS = {"Furniture": "#e69f00", "Office Supplies": "#009e73", "Technology": "#cc79a7"}
DIVERGING = "RdBu"
MARGIN_RANGE = [-30, 30]
OUTCOME_COLORS = {"Profitable": "#2b83ba", "Loss-making": "#c62828"}


def style(fig, height=420):
    ink = "#000000"
    black_font = dict(family="Inter, sans-serif", color=ink)
    axis = dict(
        linecolor=ink,
        gridcolor="#e5e5e5",
        zerolinecolor=ink,
        tickfont=black_font,
        title_font=black_font,
    )
    fig.update_layout(
        template="plotly_white",
        height=height,
        font=dict(family="Inter, sans-serif", size=12, color=ink),
        title_font=dict(family="Inter, sans-serif", size=13, color=ink),
        paper_bgcolor="#ffffff",
        plot_bgcolor="#ffffff",
        legend=dict(font=black_font, title_font=black_font),
        coloraxis_colorbar=dict(tickfont=black_font, title_font=black_font),
        hoverlabel=dict(
            bgcolor="#ffffff",
            bordercolor=ink,
            font=dict(family="Inter, sans-serif", size=12, color=ink),
        ),
        margin=dict(l=10, r=10, t=36, b=10),
    )
    fig.update_xaxes(**axis)
    fig.update_yaxes(**axis)
    fig.for_each_annotation(lambda a: a.update(font=black_font))
    return fig


def hex_to_rgba(hex_color: str, alpha: float) -> str:
    h = hex_color.lstrip("#")
    r, g, b = (int(h[i:i + 2], 16) for i in (0, 2, 4))
    return f"rgba({r},{g},{b},{alpha})"


def usd(x: float) -> str:
    sign, x = ("-" if x < 0 else ""), abs(x)
    if x >= 1e6:
        return f"{sign}{x / 1e6:.2f}M USD"
    if x >= 1e4:
        return f"{sign}{x / 1e3:.0f}K USD"
    return f"{sign}{x:,.0f} USD"


def chapter(kicker: str, title: str, question: str):
    st.markdown(
        f'<div class="chapter"><div class="kicker">{kicker}</div>'
        f'<div class="title">{title}</div><div class="question">{question}</div></div>',
        unsafe_allow_html=True,
    )


def takeaway(text: str, label: str = "So what"):
    st.markdown(f'<div class="takeaway"><b>{label}:</b> {text}</div>', unsafe_allow_html=True)


def render_part1(f: pd.DataFrame):
    chapter(
        "PART 1 OF 3",
        "Growth and Footprint",
        "Where does the money come from: over time, across countries and across products?",
    )

    with st.container(border=True):
        st.subheader("01. Monthly Sales & Profit Trajectory")
        st.caption("Use the lower range slider to isolate specific time windows.")
        g = (
            f.groupby("order_month")[["sales", "profit"]].sum().reset_index()
            .rename(columns={"sales": "Total Sales", "profit": "Total Profit"})
        )
        fig = px.line(
            g, x="order_month", y=["Total Sales", "Total Profit"], markers=True,
            color_discrete_map={"Total Sales": "#1f77b4", "Total Profit": "#d95f02"},
            labels={"order_month": "Month", "value": "Amount ($)", "variable": "Metric"},
        )
        fig.update_traces(
            line=dict(width=2.5),
            marker=dict(size=6),
            hovertemplate="<b>%{fullData.name}:</b> $%{y:,.0f}<extra></extra>",
        )
        fig.update_layout(hovermode="x unified")
        fig.update_xaxes(rangeslider_visible=True, hoverformat="%B %Y")
        st.plotly_chart(style(fig, 400), width="stretch")

        yearly = margin_by(f, "year")
        if len(yearly) > 1:
            first, last = yearly.iloc[0], yearly.iloc[-1]
            growth = (last["sales"] / first["sales"] - 1) * 100
            takeaway(
                f"Sales changed by {growth:+.0f}% between {int(first['year'])} and {int(last['year'])}, "
                f"while the profit margin moved from {first['margin']:.1f}% to {last['margin']:.1f}%."
            )

    left, right = st.columns([1.25, 1])

    with left:
        with st.container(border=True):
            head_l, head_r = st.columns([2.5, 1])
            with head_l:
                st.subheader("02. Country Performance")
                st.caption("Switch the metric to see where we sell the most and where we earn (or lose) money.")
            with head_r:
                metric = st.selectbox("Metric", ["Sales", "Profit", "Margin"], index=0)

            g = margin_by(f, "country")
            col = metric.lower()
            scale, mid = ("Blues", None) if metric == "Sales" else (DIVERGING, 0)
            fig = px.choropleth(
                g, locations="country", locationmode="country names", color=col,
                hover_name="country",
                custom_data=["sales", "profit", "margin"],
                color_continuous_scale=scale,
                color_continuous_midpoint=mid,
                range_color=MARGIN_RANGE if metric == "Margin" else None,
                labels={col: f"{metric} ({'%' if metric == 'Margin' else '$'})"},
            )
            fig.update_traces(
                hovertemplate=(
                    "<b>%{hovertext}</b><br>"
                    "Total Sales: $%{customdata[0]:,.0f}<br>"
                    "Total Profit: $%{customdata[1]:,.0f}<br>"
                    "Profit Margin: %{customdata[2]:.1f}%"
                    "<extra></extra>"
                )
            )
            fig.update_geos(showframe=True, framecolor="#000000", projection_type="natural earth")
            st.plotly_chart(style(fig, 460), width="stretch")

            top = g.sort_values(col, ascending=False).iloc[0]
            n_loss = int((g["profit"] < 0).sum())
            takeaway(
                f"Highest {metric.lower()}: {top['country']}. "
                f"{n_loss} of {len(g)} countries lose money overall."
            )

    with right:
        with st.container(border=True):
            st.subheader("03. Profit Margin by Product Hierarchy")
            st.caption("Rectangle size represents total sales; color indicates profit margin (% capped at ±30%).")
            g = margin_by(f, ["category", "sub_category"])
            fig = px.treemap(
                g, path=[px.Constant("All Products"), "category", "sub_category"],
                values="sales", color="margin", color_continuous_scale=DIVERGING,
                color_continuous_midpoint=0, range_color=MARGIN_RANGE,
                custom_data=["profit", "margin"],
                labels={"margin": "Profit Margin (%)"},
            )
            fig.update_traces(
                hovertemplate=(
                    "<b>%{label}</b><br>"
                    "Total Sales: $%{value:,.0f}<br>"
                    "Total Profit: $%{customdata[0]:,.0f}<br>"
                    "Profit Margin: %{customdata[1]:.1f}%"
                    "<extra></extra>"
                )
            )
            st.plotly_chart(style(fig, 460), width="stretch")

            ranked = g.dropna(subset=["margin"]).sort_values("margin")
            if not ranked.empty:
                worst, best = ranked.iloc[0], ranked.iloc[-1]
                takeaway(
                    f"Best margin: {best['sub_category']} ({best['margin']:.1f}%). "
                    f"Worst: {worst['sub_category']} ({worst['margin']:.1f}%)."
                )


def render_part2(f: pd.DataFrame):
    chapter(
        "PART 2 OF 3",
        "The Discount Effect",
        "Are discounts buying growth, or giving profit away?",
    )
    left, right = st.columns(2)

    with left:
        with st.container(border=True):
            st.subheader("04. Discount Brackets vs. Profit Margin")
            st.caption("Profit margin of all orders within each discount bracket (the rate).")
            g = margin_by(f, "discount_group")
            fig = px.bar(
                g, x="discount_group", y="margin", color="margin", text=g["margin"].round(1),
                color_continuous_scale=DIVERGING, color_continuous_midpoint=0,
                range_color=MARGIN_RANGE,
                category_orders={"discount_group": DISCOUNT_LABELS},
                custom_data=["sales", "profit"],
                labels={"discount_group": "Discount Bracket", "margin": "Profit Margin (%)"},
            )
            fig.update_traces(
                hovertemplate=(
                    "<b>Discount Bracket: %{x}</b><br>"
                    "Profit Margin: %{y:.1f}%<br>"
                    "Total Sales: $%{customdata[0]:,.0f}<br>"
                    "Total Profit: $%{customdata[1]:,.0f}"
                    "<extra></extra>"
                )
            )
            fig.update_coloraxes(showscale=False)
            st.plotly_chart(style(fig, 440), width="stretch")

            m = g.set_index("discount_group")["margin"]
            if len(m) > 1:
                takeaway(
                    f"Margin goes from {m.iloc[0]:.1f}% in the {m.index[0]} bracket "
                    f"to {m.iloc[-1]:.1f}% in the {m.index[-1]} bracket."
                )

    with right:
        with st.container(border=True):
            st.subheader("05. Where Discounted Revenue Ends Up")
            st.caption("Red = revenue from orders that ended in a loss.")
            s = f.assign(outcome=np.where(f["is_loss"] == 1, "Loss-making", "Profitable"))
            l1 = s.groupby(["category", "discount_group"], observed=True)["sales"].sum().reset_index()
            l2 = s.groupby(["discount_group", "outcome"], observed=True)["sales"].sum().reset_index()
            l1.columns = l2.columns = ["source", "target", "value"]
            links = pd.concat([l1, l2], ignore_index=True).astype({"source": str, "target": str})
            links = links[links["value"] > 0]

            used = set(links["source"]) | set(links["target"])
            nodes = [n for n in [*CAT_COLORS, *DISCOUNT_LABELS, *OUTCOME_COLORS] if n in used]
            idx = {n: i for i, n in enumerate(nodes)}
            node_colors = {
                **CAT_COLORS, **OUTCOME_COLORS,
                "0%": "#999999", "1%-20%": "#777777", "21%-40%": "#444444", "41%+": "#000000",
            }
            link_colors = [
                hex_to_rgba(node_colors[t if t in OUTCOME_COLORS else src], 0.4)
                for src, t in zip(links["source"], links["target"])
            ]
            fig = go.Figure(go.Sankey(
                textfont=dict(color="#000000", size=12, family="Inter, sans-serif"),
                node=dict(
                    label=nodes, color=[node_colors[n] for n in nodes], pad=15, thickness=16,
                    line=dict(width=0),
                    hovertemplate="<b>%{label}</b><br>Total Sales Volume: $%{value:,.0f}<extra></extra>",
                ),
                link=dict(
                    source=links["source"].map(idx), target=links["target"].map(idx),
                    value=links["value"], color=link_colors,
                    hovertemplate=(
                        "<b>%{source.label} → %{target.label}</b><br>"
                        "Sales Flow: $%{value:,.0f}<extra></extra>"
                    ),
                ),
            ))
            st.plotly_chart(style(fig, 440), width="stretch")

            total = s["sales"].sum()
            loss_sales = s.loc[s["is_loss"] == 1, "sales"].sum()
            if loss_sales > 0:
                deep_sales = s.loc[s["discount"] > 0.20, "sales"].sum()
                deep_loss = s.loc[(s["is_loss"] == 1) & (s["discount"] > 0.20), "sales"].sum()
                takeaway(
                    f"{loss_sales / total:.1%} of revenue comes from orders that lost money. Orders discounted "
                    f"above 20% are {deep_sales / total:.0%} of revenue but {deep_loss / loss_sales:.0%} "
                    "of that loss-making revenue."
                )

    c1, c2 = st.columns(2)
    with c1:
        with st.container(border=True):
            st.subheader("06. Discount vs. Margin Over Time")
            st.caption("Each bubble is a sub-category (size = sales). Press play to animate across years.")
            g = (
                f.groupby(["year", "category", "sub_category"], as_index=False)
                .agg(sales=("sales", "sum"), profit=("profit", "sum"), avg_discount=("discount", "mean"))
            )
            g["margin"] = np.where(g["sales"] > 0, g["profit"] / g["sales"] * 100, np.nan)
            g["avg_discount_pct"] = g["avg_discount"] * 100
            g = g.dropna(subset=["margin"]).sort_values("year")
            fig = px.scatter(
                g, x="avg_discount_pct", y="margin", size="sales", color="category",
                color_discrete_map=CAT_COLORS,
                hover_name="sub_category",
                custom_data=["category", "sales", "profit", "year"],
                animation_frame="year", animation_group="sub_category",
                range_x=[0, g["avg_discount_pct"].max() * 1.1],
                range_y=[g["margin"].min() - 5, g["margin"].max() + 5], size_max=45,
                labels={
                    "avg_discount_pct": "Average Discount (%)",
                    "margin": "Profit Margin (%)",
                    "category": "Category",
                },
            )
            bubble_hover = (
                "<b>%{hovertext}</b> (%{customdata[0]})<br>"
                "Year: %{customdata[3]}<br>"
                "Total Sales: $%{customdata[1]:,.0f}<br>"
                "Total Profit: $%{customdata[2]:,.0f}<br>"
                "Average Discount: %{x:.1f}%<br>"
                "Profit Margin: %{y:.1f}%"
                "<extra></extra>"
            )
            fig.update_traces(hovertemplate=bubble_hover)
            for frame in fig.frames:
                for trace in frame.data:
                    trace.hovertemplate = bubble_hover
            fig.add_hline(y=0, line_dash="dash", line_color="#000000")
            st.plotly_chart(style(fig, 440), width="stretch")
            takeaway(
                "Bubbles below the dashed zero-line are sub-categories where discounting has eroded profitability.",
                label="How to read",
            )

    with c2:
        with st.container(border=True):
            st.subheader("07. Multi-Dimensional Order Profile")
            st.caption("Sample of up to 3,000 orders. Drag vertically along any axis to filter ranges interactively.")
            sample = f.sample(min(len(f), 3000), random_state=42)
            lim = float(np.nan_to_num(sample["profit"].abs().quantile(0.95))) or 1.0
            fig = px.parallel_coordinates(
                sample,
                dimensions=["sales", "quantity", "discount", "shipping_cost", "profit"],
                color="profit",
                color_continuous_scale=DIVERGING,
                color_continuous_midpoint=0,
                range_color=[-lim, lim],
                labels={
                    "sales": "Sales ($)", "quantity": "Qty", "discount": "Discount",
                    "shipping_cost": "Shipping ($)", "profit": "Profit ($)",
                },
            )
            fig.update_layout(margin=dict(l=35, r=35, t=45, b=15))
            st.plotly_chart(style(fig, 440), width="stretch")
            takeaway(
                "Brushing high discount values (0.4+) immediately isolates red trajectories plunging into negative profit.",
                label="Interactive tip",
            )


def render_part3(f: pd.DataFrame, total_profit: float):
    chapter(
        "PART 3 OF 3",
        "Pinpointing the Leak",
        "Where exactly does the money disappear, and is delivery part of the problem?",
    )
    left, right = st.columns(2)

    with left:
        with st.container(border=True):
            st.subheader("08. Margin Matrix: Sub-Category vs. Market")
            st.caption("Profit margin (%) sorted worst-to-best. Red cells lose money.")
            g = margin_by(f, ["sub_category", "market"])
            pivot = g.pivot(index="sub_category", columns="market", values="margin")
            pivot = pivot.loc[
                margin_by(f, "sub_category").sort_values("margin")["sub_category"],
                margin_by(f, "market").sort_values("margin")["market"],
            ]
            orders = f.pivot_table(
                index="sub_category", columns="market", values="sales", aggfunc="size"
            ).reindex_like(pivot)
            fig = px.imshow(
                pivot, text_auto=".1f", aspect="auto", color_continuous_scale=DIVERGING,
                color_continuous_midpoint=0, range_color=MARGIN_RANGE,
                labels=dict(color="Margin (%)"),
            )
            fig.update_traces(
                customdata=orders.values,
                hovertemplate=(
                    "<b>%{y} — %{x} Market</b><br>"
                    "Profit Margin: %{z:.1f}%<br>"
                    "Orders: %{customdata:,.0f}"
                    "<extra></extra>"
                ),
            )
            st.plotly_chart(style(fig, 480), width="stretch")

            cells = pivot.stack()
            if not cells.empty:
                solid = cells[orders.stack().reindex(cells.index) >= 30]
                pool = solid if not solid.empty else cells
                (sub, mkt), val = pool.idxmin(), pool.min()
                takeaway(
                    f"Weakest cell{' with 30+ orders' if not solid.empty else ''}: {sub} in {mkt} "
                    f"({val:.1f}%). {int((cells < 0).sum())} of {len(cells)} combinations are loss-making."
                )

    with right:
        with st.container(border=True):
            st.subheader("09. Net Loss Pockets in Dollars")
            st.caption("Inner = market, middle = category, outer = sub-category. Color = average discount (%).")
            keys = ["market", "category", "sub_category"]
            pockets = margin_by(f, keys).merge(
                f.groupby(keys, as_index=False)["discount"].mean().rename(columns={"discount": "avg_discount"}),
                on=keys,
            )
            pockets = pockets[pockets["profit"] < 0].assign(
                loss=lambda d: -d["profit"], avg_discount=lambda d: d["avg_discount"] * 100
            )
            if pockets.empty:
                st.info("No loss-making pockets in the current selection.")
            else:
                fig = px.sunburst(
                    pockets, path=keys, values="loss", color="avg_discount",
                    color_continuous_scale="Reds", labels={"avg_discount": "Avg Discount (%)"},
                )
                fig.update_traces(
                    insidetextorientation="radial",
                    textfont=dict(color="#000000", family="Inter, sans-serif"),
                    marker=dict(line=dict(color="#ffffff", width=1)),
                    hovertemplate=(
                        "<b>%{label}</b><br>"
                        "Path: %{id}<br>"
                        "Net Loss: $%{value:,.0f}<br>"
                        "Share of Parent: %{percentParent:.1%}<br>"
                        "Avg Discount: %{color:.1f}%"
                        "<extra></extra>"
                    ),
                )
                st.plotly_chart(style(fig, 480), width="stretch")

                total_loss = pockets["loss"].sum()
                biggest = pockets.sort_values("loss", ascending=False).iloc[0]
                text = (
                    f"{len(pockets)} pockets lose {usd(total_loss)} on net. Biggest: "
                    f"{biggest['sub_category']} in {biggest['market']} ({biggest['loss'] / total_loss:.0%} of it)."
                )
                if total_profit > 0:
                    text += f" If they only broke even, total profit would rise by {total_loss / total_profit:.0%}."
                takeaway(text)

    with st.container(border=True):
        st.subheader("10. Delivery Lead Time by Ship Mode & Priority")
        st.caption("Box = IQR, solid line = median days, dashed line = average days.")
        ship_order = ["Same Day", "First Class", "Second Class", "Standard Class"]
        prio_order = ["Critical", "High", "Medium", "Low"]
        fig = px.box(
            f, x="ship_mode", y="delivery_days", color="order_priority", points=False,
            category_orders={"ship_mode": ship_order, "order_priority": prio_order},
            color_discrete_map={
                "Critical": "#b2182b", "High": "#e66101", "Medium": "#2b83ba", "Low": "#111111",
            },
            labels={
                "ship_mode": "Shipping Mode (Fastest → Slowest)",
                "delivery_days": "Delivery Time",
                "order_priority": "Priority",
            },
        )
        fig.update_traces(boxmean=True, line=dict(width=1.8))
        fig.update_yaxes(dtick=1, ticksuffix=" d")
        fig.update_layout(
            boxgap=0.25,
            boxgroupgap=0.15,
            legend=dict(
                orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0,
                title_text="Order Priority:",
            ),
        )
        st.plotly_chart(style(fig, 420), width="stretch")

        med = f.groupby("ship_mode")["delivery_days"].median()
        takeaway(
            "Median delivery time — "
            + " · ".join(f"{m}: {med[m]:.0f} d" for m in ship_order if m in med.index),
            label="Reading",
        )


def render_machine_learning(f: pd.DataFrame, df_full: pd.DataFrame, all_cats: list, all_markets: list):
    chapter(
        "FINALE · 2 ML MODELS · ALL TOGETHER",
        "From Description to Prediction",
        "Can we group our customers, and flag orders that will lose money before they ship?",
    )
    cust = f.groupby("customer_id").agg(
        total_sales=("sales", "sum"), total_profit=("profit", "sum"),
        avg_discount=("discount", "mean"), n_orders=("order_id", "nunique"),
        avg_shipping=("shipping_cost", "mean"),
    )
    left, right = st.columns([1.3, 1])

    with left:
        with st.container(border=True):
            head_l, head_r = st.columns([2.5, 1])
            with head_l:
                st.subheader("ML 01. K-Means Customer Segmentation (PCA Projection)")
                st.caption("Customers clustered by spend, profit, discount, order count and shipping cost.")
            with head_r:
                k = st.slider("Clusters (k)", 2, 8, 4)

            seg = segment_customers(cust, k) if len(cust) > k else None
            if seg is None:
                st.info("Not enough customers for this many clusters with the current filters.")
            else:
                fig = px.scatter(
                    seg.reset_index(), x="PC1", y="PC2", color="cluster",
                    category_orders={"cluster": sorted(seg["cluster"].unique(), key=int)},
                    color_discrete_sequence=[
                        "#000000", "#e66101", "#2b83ba", "#5e3c99",
                        "#1b9e77", "#d95f02", "#7570b3", "#666666",
                    ],
                    custom_data=[
                        "customer_id", "total_sales", "total_profit", "avg_discount_pct",
                        "n_orders", "avg_shipping", "cluster",
                    ],
                    labels={"cluster": "Cluster"},
                )
                fig.update_traces(
                    hovertemplate=(
                        "<b>Customer ID: %{customdata[0]}</b> (Cluster %{customdata[6]})<br>"
                        "Total Sales: $%{customdata[1]:,.0f}<br>"
                        "Total Profit: $%{customdata[2]:,.0f}<br>"
                        "Average Discount: %{customdata[3]:.1f}%<br>"
                        "Total Orders: %{customdata[4]}<br>"
                        "Avg. Shipping Cost: $%{customdata[5]:,.2f}"
                        "<extra></extra>"
                    )
                )
                st.plotly_chart(style(fig, 400), width="stretch")

    with right:
        with st.container(border=True):
            st.subheader("Cluster Profile Summary")
            st.caption("Mean customer metrics per segment.")
            if seg is not None:
                profile = seg.groupby("cluster").agg(
                    customers=("total_sales", "size"),
                    avg_sales=("total_sales", "mean"),
                    avg_profit=("total_profit", "mean"),
                    avg_discount_pct=("avg_discount_pct", "mean"),
                    n_orders=("n_orders", "mean"),
                    avg_shipping=("avg_shipping", "mean"),
                ).round(2)
                st.dataframe(profile, width="stretch", height=330)
                weakest = profile["avg_profit"].idxmin()
                row = profile.loc[weakest]
                takeaway(
                    f"Cluster {weakest} is the least profitable segment: {usd(row['avg_profit'])} "
                    f"average profit per customer at a {row['avg_discount_pct']:.0f}% average discount."
                )

    pipe, scores, cm, imp = train_model(df_full)
    rf_l, rf_r = st.columns([1.35, 1])

    with rf_l:
        with st.container(border=True):
            st.subheader("ML 02. Random Forest Loss Classifier")
            st.caption(
                f"Test accuracy {scores['accuracy']:.1%} (baseline {scores['baseline']:.1%})  |  "
                f"ROC AUC {scores['auc']:.3f}  |  Loss recall {scores['loss_recall']:.1%}  |  20% holdout"
            )
            sub_l, sub_r = st.columns(2)
            top = imp.sort_values().tail(10) * 100
            top.index = [
                i.replace("cat__", "").replace("remainder__", "").replace("_", " ").title()
                for i in top.index
            ]
            fig_imp = px.bar(
                top, orientation="h",
                labels={"value": "Importance (%)", "index": "Feature"},
                title="Top 10 Feature Importances",
            )
            fig_imp.update_traces(
                marker_color="#000000",
                hovertemplate="<b>%{y}</b><br>Relative Importance: %{x:.1f}%<extra></extra>",
            )
            fig_imp.update_layout(showlegend=False)
            sub_l.plotly_chart(style(fig_imp, 350), width="stretch")

            fig_cm = px.imshow(
                cm, text_auto=True,
                x=["Predicted: Profit", "Predicted: Loss"],
                y=["Actual: Profit", "Actual: Loss"],
                color_continuous_scale="Greys",
                title="Confusion Matrix",
            )
            fig_cm.update_traces(
                hovertemplate="<b>%{y} | %{x}</b><br>Order Count: %{z:,}<extra></extra>"
            )
            sub_r.plotly_chart(style(fig_cm, 350), width="stretch")

            takeaway(
                f"The strongest predictor of a loss-making order is {top.index[-1]} "
                f"({top.iloc[-1]:.1f}% of total importance)."
            )

    with rf_r:
        with st.container(border=True):
            st.subheader("What-If Order Loss Simulator")
            st.caption("Configure an order to estimate its probability of resulting in a net loss.")
            c_a, c_b = st.columns(2)
            w_disc = c_a.slider("Discount", 0.0, 0.85, 0.2, 0.05)
            w_cat = c_b.selectbox("Category", all_cats)
            c_c, c_d = st.columns(2)
            w_mkt = c_c.selectbox("Market", all_markets)
            w_ship = c_d.selectbox("Ship Mode", sorted(df_full["ship_mode"].unique()))

            row = pd.DataFrame([{
                "discount": w_disc,
                "sales": df_full["sales"].median(),
                "quantity": df_full["quantity"].median(),
                "shipping_cost": df_full["shipping_cost"].median(),
                "delivery_days": df_full["delivery_days"].median(),
                "category": w_cat, "market": w_mkt, "segment": "Consumer", "ship_mode": w_ship,
            }])
            p = float(pipe.predict_proba(row[NUM + CAT])[0, 1])

            if p < 0.30:
                risk, bar_color = "LOW RISK", "#2b83ba"
            elif p < 0.60:
                risk, bar_color = "MEDIUM RISK", "#e69f00"
            else:
                risk, bar_color = "HIGH RISK", "#c62828"

            gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=p * 100,
                number={"suffix": "%", "font": {"size": 38, "color": "#000000", "family": "Inter, sans-serif"}},
                title={"text": f"LOSS PROBABILITY · <b>{risk}</b>", "font": {"size": 12, "color": "#000000"}},
                gauge={
                    "axis": {"range": [0, 100], "ticksuffix": "%", "tickcolor": "#000000"},
                    "bar": {"color": bar_color, "thickness": 0.32},
                    "bgcolor": "#ffffff",
                    "borderwidth": 1,
                    "bordercolor": "#000000",
                    "steps": [
                        {"range": [0, 30], "color": "#eef5fa"},
                        {"range": [30, 60], "color": "#fdf5e6"},
                        {"range": [60, 100], "color": "#fbeaea"},
                    ],
                },
            ))
            gauge.update_layout(
                height=230,
                margin=dict(l=20, r=20, t=40, b=10),
                paper_bgcolor="#ffffff",
                font=dict(family="Inter, sans-serif", color="#000000"),
            )
            st.plotly_chart(gauge, width="stretch")