import seaborn as sns; sns.set(color_codes=True)

sns.set(rc={'figure.figsize':(10, 4)})
sns.set_style("white")
for i, (name, seq) in enumerate(seq_lens.items()):
    seq.sort()
    data = pd.DataFrame(data=[[i, s] for (i, s) in enumerate(seq)], columns=["Problems Solved", "Number of Steps"])

    ax = sns.regplot(x="Problems Solved", y="Number of Steps", data=data, label=name, ci=None,
                     scatter_kws={"color": colors[i], "s":3}, line_kws={"color": "#263f44", "ls": "dashed", "lw":1})
ax.set_yscale('log')
ax.legend()
