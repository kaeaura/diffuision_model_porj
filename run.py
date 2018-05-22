import os
import sys
import getopt
import networkx as nx
import diffusion_models as dm

def main(argv):
	g_path = ""
	p_path = ""
	h_path = ""
	o_path = ""
	model = "pagerank"
	alpha = 0.85
	gamma = 1
	s_path = ""
	heat = []
	seeds = []

	try:
		opts, args = getopt.getopt(argv, "hG:P:H:O:plcda:r:s:", ["help", "graph=", "pref=", "heat=", "alpha=", "gamma=", "seeds="])
	except getopt.GetoptError:
		print "The given arguments incorrect"
		sys.exit(2)

	for opt, arg in opts:
		if opt in ("-h", "--help"):
			print ("-h, --help : show help")
			print ("-G, --graph [file_path] : file contains the edge list of a given graph")
			print ("-P, --pref [file_path] : preference path")
			print ("-H, --heat [file_path] : heat vector (for DiffusionRank only)")
			print ("-O [file_path] : output path")
			print ("-p : set model as PageRank")
			print ("-l : set model as Linear Threshold Model")
			print ("-c : set model as Independent Cascade Model")
			print ("-d : set model as DiffusionRank")
			print ("-a, --alpha [float] : parameter alpha (damping factor) in PageRank")
			print ("-r, --gamma [float] : parameter gamma (or time) in DiffusionRank")
			print ("-s, --seeds [file_path] : parameter seeds in Linear Threshold Model")
			sys.exit()
		elif opt in ("-G", "--graph"):
			g_path = arg
		elif opt in ("-P", "--pref"):
			p_path = arg
		elif opt in ("-H", "--heat"):
			h_path = arg
		elif opt == "-O":
			o_path = arg
		elif opt == "-p":
			model = "page_rank"
		elif opt == "-l":
			model = "linear_threshold"
		elif opt == "-c":
			model = "independent_cascade"
		elif opt == "-d":
			model = "diffusion_rank"
		elif opt in ("-a", "--alpha"):
			alpha = float(arg)
		elif opt in ("-r", "--gamma"):
			gamma = float(arg)
		elif opt in ("-s", "--seeds"):
			s_path = arg

	g = nx.read_edgelist(g_path, create_using=nx.DiGraph(), nodetype = int)

	if model == "page_rank":
		values = nx.pagerank(g, alpha = 0.85).values()
	elif model == "diffusion_rank":
		try:
			with open(h_path, 'r') as f:
				for l in f.readlines():
					heat.append(float(l.strip()))
		except:
			raise
		values = dm.diffusionrank(g, heat, gamma = gamma).getA1().tolist()
	elif model == "linear_threshold":
		try:
			with open(s_path, 'r') as f:
				for l in f.readlines():
					seeds.append(int(l.strip()))
		except:
			raise
		values = dm.linearthreshold(g, seeds).values()
	elif model == "independent_cascade":
		try:
			with open(s_path, 'r') as f:
				for l in f.readlines():
					seeds.append(int(l.strip()))
		except:
			raise
		values = dm.independentcascade(g, seeds)
	else:
		print ("The model does not implemented")
		sys.exit()

	if not os.path.exists(os.path.dirname(o_path)):
		try:
			os.mkdir(os.path.dirname(o_path))
		except os.error:
			pass

	with open(o_path, "w") as f:
		for value in values:
			f.write("%f\n" % value)

if __name__ == "__main__":
	if len(sys.argv) > 1:
		main(sys.argv[1:])
	else:
		print ("need to input args, please see --help")
