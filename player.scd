(
s.waitForBoot({
    var slicesDir = "/Users/vkelemen/workspace/sample-slicer/slices";
    var numGroups = 4;

    // Find available sample sets
    var sampleSets = (slicesDir +/+ "*").pathMatch.select({ arg p; PathName(p).isFolder }).collect({ arg p; PathName(p).folderName });

    // Euclidean rhythm generator
    var euclidean = { arg hits, steps;
        var pattern = Array.fill(steps, 0);
        var bucket = 0;
        steps.do({ arg i;
            bucket = bucket + hits;
            if(bucket >= steps, {
                bucket = bucket - steps;
                pattern[i] = 1;
            });
        });
        pattern;
    };

    // Pattern to string
    var patternString = { arg pat;
        var chars = [". ", "x "];
        var str = "Pattern: ";
        pat.do({ arg x; str = str ++ chars[x] });
        str;
    };

    // Load samples for a given set
    var loadSampleSet = { arg setName;
        var basePath = slicesDir +/+ setName;
        ~groupBufs.do({ arg bufs; bufs.do({ arg b; b.free }) });
        ~groupBufs = numGroups.collect({ arg i;
            var path = basePath +/+ "group_" ++ i;
            var files = (path +/+ "*.wav").pathMatch;
            files.collect({ arg f; Buffer.read(s, f) });
        });
        ~groupNames = numGroups.collect({ arg i;
            var path = basePath +/+ "group_" ++ i;
            var files = (path +/+ "*.wav").pathMatch;
            files.collect({ arg f; f.basename });
        });
        ("Loaded: " ++ setName).postln;
        numGroups.do({ arg i;
            ("  Group " ++ i ++ ": " ++ ~groupBufs[i].size ++ " samples").postln;
        });
    };

    SynthDef(\sampler, {
        arg buf = 0, amp = 0.5, gate = 1;
        var sig = PlayBuf.ar(2, buf, BufRateScale.kr(buf), doneAction: 0);
        var done = Done.kr(sig);
        var env = EnvGen.kr(Env.asr(0.03, 1, 0.03, \sin), gate - done, doneAction: 2);
        Out.ar(0, sig * amp * env);
    }).add;

    s.sync;

    // Initialize
    ~groupBufs = Array.new;
    ~groupNames = Array.new;
    ~currentGroup = 0;
    ~currentSet = 0;
    ~synth = nil;
    ~running = false;

    // Euclidean params
    ~hits = 3;
    ~steps = 8;
    ~stepDur = 0.25;

    // Load first sample set
    loadSampleSet.value(sampleSets[0]);

    s.sync;

    ~win = Window("Sample Player", Rect(100, 100, 320, 300)).front;
    ~win.background = Color.gray(0.2);

    // Sample set selector
    StaticText(~win, Rect(20, 10, 80, 20))
        .string_("Samples:")
        .stringColor_(Color.white)
        .font_(Font("Monaco", 12));

    PopUpMenu(~win, Rect(100, 10, 200, 20))
        .items_(sampleSets)
        .value_(0)
        .action_({ arg menu;
            ~currentSet = menu.value;
            ~running = false;
            try { ~synth.set(\gate, 0) };
            { ~label.string = "--" }.defer;
            fork {
                0.1.wait;
                loadSampleSet.value(sampleSets[~currentSet]);
            };
        });

    // Group selector
    StaticText(~win, Rect(20, 45, 260, 20))
        .string_("Group:")
        .stringColor_(Color.white)
        .font_(Font("Monaco", 12));

    ~groupBtns = numGroups.collect({ arg i;
        Button(~win, Rect(20 + (i * 70), 65, 65, 30))
            .states_([
                [i.asString, Color.white, Color.gray(0.4)],
                [i.asString, Color.black, Color.cyan]
            ])
            .action_({ arg btn;
                ~currentGroup = i;
                ~groupBtns.do({ arg b; b.value = 0 });
                btn.value = 1;
            });
    });
    ~groupBtns[0].value = 1;

    // Euclidean controls
    StaticText(~win, Rect(20, 110, 60, 20))
        .string_("Hits:")
        .stringColor_(Color.white)
        .font_(Font("Monaco", 12));

    ~hitsBox = NumberBox(~win, Rect(80, 110, 50, 20))
        .value_(~hits)
        .decimals_(0)
        .clipLo_(1)
        .clipHi_(16);

    StaticText(~win, Rect(150, 110, 60, 20))
        .string_("Steps:")
        .stringColor_(Color.white)
        .font_(Font("Monaco", 12));

    ~stepsBox = NumberBox(~win, Rect(210, 110, 50, 20))
        .value_(~steps)
        .decimals_(0)
        .clipLo_(1)
        .clipHi_(16);

    StaticText(~win, Rect(20, 140, 100, 20))
        .string_("Step (sec):")
        .stringColor_(Color.white)
        .font_(Font("Monaco", 12));

    ~durBox = NumberBox(~win, Rect(120, 140, 60, 20))
        .value_(~stepDur)
        .decimals_(2)
        .clipLo_(0.05)
        .clipHi_(2.0)
        .action_({ arg box; ~stepDur = box.value });

    // Pattern display
    ~patternLabel = StaticText(~win, Rect(20, 170, 280, 20))
        .string_(patternString.value(euclidean.value(~hits, ~steps)))
        .stringColor_(Color.yellow)
        .font_(Font("Monaco", 12));

    // Update pattern display when params change
    ~updatePattern = {
        var pat = euclidean.value(~hits, ~steps);
        { ~patternLabel.string = patternString.value(pat) }.defer;
    };

    ~hitsBox.action = { arg box; ~hits = box.value.asInteger; ~updatePattern.value };
    ~stepsBox.action = { arg box; ~steps = box.value.asInteger; ~updatePattern.value };

    // Sample name label
    ~label = StaticText(~win, Rect(20, 200, 280, 30));
    ~label.string = "--";
    ~label.stringColor = Color.green;
    ~label.font = Font("Monaco", 14);

    // Play button
    Button(~win, Rect(20, 240, 120, 40))
        .states_([["PLAY", Color.white, Color.green(0.6)]])
        .action_({
            if(~running.not, {
                ~running = true;
                fork {
                    while({ ~running }, {
                        var pattern = euclidean.value(~hits, ~steps);
                        pattern.do({ arg hit;
                            if(~running, {
                                if(hit == 1, {
                                    var bufs = ~groupBufs[~currentGroup];
                                    var names = ~groupNames[~currentGroup];
                                    var i = bufs.size.rand;
                                    try { ~synth.set(\gate, 0) };
                                    ~synth = Synth(\sampler, [\buf, bufs[i], \amp, 0.5]);
                                    { ~label.string = names[i] }.defer;
                                });
                                ~stepDur.wait;
                            });
                        });
                    });
                };
            });
        });

    // Stop button
    Button(~win, Rect(160, 240, 120, 40))
        .states_([["STOP", Color.white, Color.red(0.6)]])
        .action_({
            ~running = false;
            try { ~synth.set(\gate, 0) };
            { ~label.string = "--" }.defer;
        });

    ~win.onClose = {
        ~running = false;
        ~groupBufs.do({ arg bufs; bufs.do({ arg b; b.free }) });
    };
});
)
