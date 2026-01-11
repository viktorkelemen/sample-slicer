(
s.waitForBoot({
    var path = "/Users/vkelemen/workspace/sample-slicer/slices/mark_fell_mix/group_1";
    var files = (path +/+ "*.wav").pathMatch;

    SynthDef(\sampler, {
        arg buf = 0, amp = 0.5, gate = 1;
        var sig = PlayBuf.ar(2, buf, BufRateScale.kr(buf), doneAction: 2);
        var env = EnvGen.kr(Env.asr(0.03, 1, 0.03, \sin), gate, doneAction: 2);
        Out.ar(0, sig * amp * env);
    }).add;

    s.sync;

    ~bufs = files.collect({ arg f; Buffer.read(s, f) });
    ~names = files.collect({ arg f; f.basename });
    ~synth = nil;
    ~running = false;

    s.sync;

    ("Loaded " ++ ~bufs.size ++ " samples").postln;

    ~win = Window("Player", Rect(100, 100, 300, 120)).front;
    ~win.background = Color.gray(0.2);

    ~label = StaticText(~win, Rect(20, 10, 260, 30));
    ~label.string = "--";
    ~label.stringColor = Color.green;
    ~label.font = Font("Monaco", 14);

    Button(~win, Rect(20, 50, 100, 40))
        .states_([["PLAY", Color.white, Color.green(0.6)]])
        .action_({
            if(~running.not, {
                ~running = true;
                fork {
                    while({ ~running }, {
                        var i = ~bufs.size.rand;
                        try { ~synth.set(\gate, 0) };
                        ~synth = Synth(\sampler, [\buf, ~bufs[i], \amp, 0.5]);
                        { ~label.string = ~names[i] }.defer;
                        2.wait;
                    });
                };
            });
        });

    Button(~win, Rect(130, 50, 100, 40))
        .states_([["STOP", Color.white, Color.red(0.6)]])
        .action_({
            ~running = false;
            try { ~synth.set(\gate, 0) };
            { ~label.string = "--" }.defer;
        });

    ~win.onClose = {
        ~running = false;
        ~bufs.do({ arg b; b.free });
    };
});
)
